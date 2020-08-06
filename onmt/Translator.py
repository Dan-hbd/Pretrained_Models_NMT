import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from onmt.ModelConstructor import build_model, build_language_model
from ae.Autoencoder import Autoencoder
import torch.nn.functional as F
import sys

model_list = ['transformer', 'stochastic_transformer', 'fusion_network']


class Translator(object):
    def __init__(self, opt):

        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self.beta = opt.beta
        self.alpha = opt.alpha
        self.start_with_bos = opt.start_with_bos
        self.fp16 = opt.fp16
        self.attributes = opt.attributes  # attributes split by |. for example: de|domain1
        self.bos_token = opt.bos_token

        self.sampling = opt.sampling

        if self.attributes:
            self.attributes = self.attributes.split("|")

        self.models = list()
        self.model_types = list()

        # models are string with | as delimiter
        models = opt.model.split("|")

        # print(models)
        self.n_models = len(models)
        self._type = 'text'

        for i, model in enumerate(models):
            model_path=model
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)

            model_opt = checkpoint['opt']

            if i == 0:
                if "src" in checkpoint['dicts']:
                    self.src_dict = checkpoint['dicts']['src']
                else:
                    self._type = "audio"
                self.tgt_dict = checkpoint['dicts']['tgt']

                if "atb" in checkpoint["dicts"]:
                    self.atb_dict = checkpoint['dicts']['atb']

                else:
                    self.atb_dict = None

                self.bos_id = self.tgt_dict.labelToIdx[self.bos_token]

            model_opt.enc_not_load_state = True
            model_opt.dec_not_load_state = True
            model = build_model(model_opt, checkpoint['dicts'])
            
            # by me
            print('Loading model from %s' % model_path)
            model.load_state_dict(checkpoint['model'])

            if model_opt.model in model_list:
                # if model.decoder.positional_encoder.len_max < self.opt.max_sent_length:
                #     print("Not enough len to decode. Renewing .. ")
                #     model.decoder.renew_buffer(self.opt.max_sent_length)
                model.renew_buffer(self.opt.max_sent_length)

            if opt.fp16:
                model = model.half()

            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()

            model.eval()

            self.models.append(model)
            self.model_types.append(model_opt.model)

        # language model
        if opt.lm is not None:
            if opt.verbose:
                print('Loading language model from %s' % opt.lm)

            lm_chkpoint = torch.load(opt.lm, map_location=lambda storage, loc: storage)

            lm_opt = lm_chkpoint['opt']

            lm_model = build_language_model(lm_opt, checkpoint['dicts'])

            if opt.fp16:
                lm_model = lm_model.half()

            if opt.cuda:
                lm_model = lm_model.cuda()
            else:
                lm_model = lm_model.cpu()

            self.lm_model = lm_model

        self.cuda = opt.cuda
        self.ensemble_op = opt.ensemble_op

        if opt.autoencoder is not None:
            if opt.verbose:
                print('Loading autoencoder from %s' % opt.autoencoder)
            checkpoint = torch.load(opt.autoencoder,
                                    map_location=lambda storage, loc: storage)
            model_opt = checkpoint['opt']

            # posSize= checkpoint['autoencoder']['nmt.decoder.positional_encoder.pos_emb'].size(0)
            # self.models[0].decoder.renew_buffer(posSize)
            # self.models[0].decoder.renew_buffer(posSize)

            # Build model from the saved option
            self.autoencoder = Autoencoder(self.models[0], model_opt)

            self.autoencoder.load_state_dict(checkpoint['autoencoder'])

            if opt.cuda:
                self.autoencoder = self.autoencoder.cuda()
                self.models[0] = self.models[0].cuda()
            else:
                self.autoencoder = self.autoencoder.cpu()
                self.models[0] = self.models[0].cpu()

            self.models[0].autoencoder = self.autoencoder
        if opt.verbose:
            print('Done')

    def init_beam_accum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    # Combine distributions from different models
    def _combine_outputs(self, outputs):

        if len(outputs) == 1:
            return outputs[0]

        if self.ensemble_op == "logSum":
            output = (outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])

            output.div_(len(outputs))

            # output = torch.log(output)
            output = F.log_softmax(output, dim=-1)
        elif self.ensemble_op == "mean":
            output = torch.exp(outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])

            output.div_(len(outputs))
            # output = torch.log(output)
            output = torch.log(output)
        elif self.ensemble_op == "max":
            output = outputs[0]

            for i in range(1, len(outputs)):
                output = torch.max(output,outputs[i])

        elif self.ensemble_op == "min":
            output = outputs[0]

            for i in range(1, len(outputs)):
                output = torch.min(output,outputs[i])

        elif self.ensemble_op == 'gmean':
            output = torch.exp(outputs[0])

            # geometric mean of the probabilities
            for i in range(1, len(outputs)):
                output *= torch.exp(outputs[i])

            # have to normalize
            output.pow_(1.0 / float(len(outputs)))
            norm_ = torch.norm(output, p=1, dim=-1)
            output.div_(norm_.unsqueeze(-1))

            output = torch.log(output)
        else:
            raise ValueError(
                'Emsemble operator needs to be "mean" or "logSum", the current value is %s' % self.ensemble_op)
        return output

    # Take the average of attention scores
    def _combine_attention(self, attns):

        attn = attns[0]

        for i in range(1, len(attns)):
            attn += attns[i]

        attn.div(len(attns))

        return attn

    def build_data(self, src_sents, tgt_sents, type='mt'):
        # This needs to be the same as preprocess.py.

        if type == 'mt':
            if self.start_with_bos:
                src_data = [self.src_dict.convertToIdx(b,
                                                       onmt.Constants.SRC_UNK_WORD,
                                                       onmt.Constants.SRC_BOS_WORD)
                            for b in src_sents]
            else:
                # src_data = [self.src_dict.convertToIdx(b,
                #                                        onmt.Constants.UNK_WORD)
                #             for b in src_sents]

                # by me
                src_data = [self.src_dict.convertToIdx(b,
                                                       onmt.Constants.SRC_UNK_WORD)
                            for b in src_sents]
        elif type == 'asr':
            # no need to deal with this
            src_data = src_sents
        else:
            raise NotImplementedError

        tgt_bos_word = self.opt.bos_token
        if self.opt.no_bos_gold:
            tgt_bos_word = None
        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.Constants.TGT_UNK_WORD,
                                                   tgt_bos_word,
                                                   onmt.Constants.TGT_EOS_WORD) for b in tgt_sents]

            # by me
            # 更新，就用他以前的
            # tgt_data = [self.tgt_dict.convertToIdx(b,
            #                                        onmt.Constants.BERT_UNK_WORD,
            #                                        tgt_bos_word,
            #                                        onmt.Constants.BERT_SEP_WORD) for b in tgt_sents]


        src_atbs = None

        if self.attributes:
            tgt_atbs = dict()

            idx = 0
            for i in self.atb_dict:
                tgt_atbs[i] = [self.atb_dict[i].convertToIdx([self.attributes[idx]], onmt.Constants.UNK_WORD)
                               for _ in src_sents]
                idx = idx + 1

        else:
            tgt_atbs = None

        return onmt.Dataset(src_data, tgt_data,
                            src_atbs=src_atbs, tgt_atbs=tgt_atbs,
                            batch_size_words=sys.maxsize,
                            data_type=self._type,
                            batch_size_sents=self.opt.batch_size)

    def build_asr_data(self, src_data, tgt_sents):
        # This needs to be the same as preprocess.py.

        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.Constants.TGT_UNK_WORD,
                                                   onmt.Constants.TGT_BOS_WORD,
                                                   onmt.Constants.TGT_EOS_WORD) for b in tgt_sents]

        return onmt.Dataset(src_data, tgt_data,
                            batch_size_words=sys.maxsize,
                            data_type=self._type, batch_size_sents=self.opt.batch_size)

    def build_target_tokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.TGT_EOS)

        # by me
        # 更新 就用以前的
        # tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.BERT_SEP)
        tokens = tokens[:-1]  # EOS

        return tokens

    def translate_batch(self, batch):

        torch.set_grad_enabled(False)
        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        batch_size = batch.size

        gold_scores = batch.get('source').data.new(batch_size).float().zero_()
        gold_words = 0
        allgold_scores = []

        if batch.has_target:
            # Use the first model to decode
            model_ = self.models[0]

            gold_words, gold_scores, allgold_scores = model_.decode(batch)

        #  (3) Start decoding

        # time x batch * beam

        # initialize the beam
        beam = [onmt.Beam(beam_size, self.bos_id, self.opt.cuda, self.opt.sampling) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        decoder_states = dict()

        for i in range(self.n_models):
            decoder_states[i] = self.models[i].create_decoder_state(batch, beam_size)

        if self.opt.lm:
            lm_decoder_states = self.lm_model.create_decoder_state(batch, beam_size)

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.

            # input size: 1 x ( batch * beam )
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)

            decoder_input = input

            # require batch first for everything
            outs = dict()
            attns = dict()

            for k in range(self.n_models):
                # decoder_hidden, coverage = self.models[k].decoder.step(decoder_input.clone(), decoder_states[k])

                # run decoding on the model
                decoder_output = self.models[k].step(decoder_input.clone(), decoder_states[k])

                # extract the required tensors from the output (a dictionary)
                outs[k] = decoder_output['log_prob']
                attns[k] = decoder_output['coverage']

            # for ensembling models
            out = self._combine_outputs(outs)
            attn = self._combine_attention(attns)

            # for lm fusion
            if self.opt.lm:
                lm_decoder_output = self.lm_model.step(decoder_input.clone(), lm_decoder_states)

                # fusion
                lm_out =  lm_decoder_output['log_prob']
                # out = out + 0.3 * lm_out

                out = lm_out
            word_lk = out.view(beam_size, remaining_sents, -1) \
                .transpose(0, 1).contiguous()
            attn = attn.view(beam_size, remaining_sents, -1) \
                .transpose(0, 1).contiguous()

            active = []

            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], attn.data[idx]):
                    active += [b]

                for j in range(self.n_models):
                    decoder_states[j].update_beam(beam, b, remaining_sents, idx)

                if self.opt.lm:
                    lm_decoder_states.update_beam(beam, b, remaining_sents, idx)

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            for j in range(self.n_models):
                decoder_states[j].prune_complete_beam(active_idx, remaining_sents)

            if self.opt.lm:
                lm_decoder_states.prune_complete_beam(active_idx, remaining_sents)

            remaining_sents = len(active)

        #  (4) package everything up
        all_hyp, all_scores, all_attn = [], [], []
        n_best = self.opt.n_best
        all_lengths = []

        for b in range(batch_size):
            scores, ks = beam[b].sortBest()

            all_scores += [scores[:n_best]]
            hyps, attn, length = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
            all_lengths += [length]
            # if(src_data.data.dim() == 3):
            if self.opt.encoder_type == 'audio':
                valid_attn = decoder_states[0].original_src.narrow(2, 0, 1).squeeze(2)[:, b].ne(onmt.Constants.PAD) \
                    .nonzero().squeeze(1)
            else:
                valid_attn = decoder_states[0].original_src[:, b].ne(onmt.Constants.PAD) \
                    .nonzero().squeeze(1)
            attn = [a.index_select(1, valid_attn) for a in attn]
            all_attn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                                                     ["%4f" % s for s in t.tolist()]
                                                     for t in beam[b].all_scores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])

        torch.set_grad_enabled(True)

        return all_hyp, all_scores, all_attn, all_lengths, gold_scores, gold_words, allgold_scores

    def translate(self, src_data, tgt_data):
        #  (1) convert words to indexes
        dataset = self.build_data(src_data, tgt_data)
        batch = dataset.next()[0]
        if self.cuda:
            batch.cuda(fp16=self.fp16)
        batch_size = batch.size

        #  (2) translate
        pred, pred_score, attn, pred_length, gold_score, gold_words, allgold_words = self.translate_batch(batch)

        #  (3) convert indexes to words
        pred_batch = []
        for b in range(batch_size):
            pred_batch.append(
                [self.build_target_tokens(pred[b][n], src_data[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return pred_batch, pred_score, pred_length, gold_score, gold_words, allgold_words

    def translate_asr(self, src_data, tgt_data):
        #  (1) convert words to indexes
        dataset = self.build_asr_data(src_data, tgt_data)
        # src, tgt = batch
        batch = dataset.next()[0]
        if self.cuda:
            batch.cuda(fp16=self.fp16)
        batch_size = batch.size

        #  (2) translate
        pred, pred_score, attn, pred_length, gold_score, gold_words, allgold_words = self.translate_batch(batch)

        #  (3) convert indexes to words
        pred_batch = []
        for b in range(batch_size):
            pred_batch.append(
                [self.build_target_tokens(pred[b][n], src_data[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return pred_batch, pred_score, pred_length, gold_score, gold_words, allgold_words


