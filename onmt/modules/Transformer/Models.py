import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, \
    PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout, embedded_dropou_bert, switchout
from torch.utils.checkpoint import checkpoint
from collections import defaultdict


torch_version = float(torch.__version__[:3])


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output

    return custom_forward


class MixedEncoder(nn.Module):

    def __init(self, text_encoder, audio_encoder):
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        if input.dim() == 2:
            return self.text_encoder.forward(input)
        else:
            return self.audio_encoder.forward(input)


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)

    """

    def __init__(self, opt, embeddings, positional_encoder, encoder_type='text'):

        super(TransformerEncoder, self).__init__()

        # # by me
        # assert bert_embeddings is not None

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.enc_hidden_dropout = opt.transformer_hidden_dropout
        self.enc_attn_dropout = opt.transformer_attn_dropout
        self.enc_emb_dropout = opt.transformer_emb_dropout
        if not opt.get_context_emb:
            self.enc_word_dropout = opt.transformer_word_dropout
        self.time = opt.time
        self.version = opt.version
        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.fp16 = opt.fp16

        # disable word dropout when switch out is in action
        if self.switchout > 0.0:
            self.enc_word_dropout = 0.0

        feature_size = opt.input_size
        self.channels = 1  # n. audio channels

        self.word_lut = embeddings 

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.enc_emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.positional_encoder = positional_encoder

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList(
            [EncoderLayer(self.n_heads, self.model_size, self.enc_hidden_dropout, self.inner_size,
                          self.enc_attn_dropout, variational=self.varitional_dropout) for _ in
             range(self.layers)])


    def forward(self, src, contextul_emb=None, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            # by me
            mask_src = src.eq(onmt.Constants.SRC_PAD).unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting
            if  contextul_emb :
                print("Use context-aware embedding in the model")
                emb = contextul_emb
            else:
                emb = embedded_dropout(self.word_lut, src, dropout=self.enc_word_dropout if self.training else 0)
        else:
            raise NotImplementedError

        if torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        # B x T x H -> T x B x H
        context = emb.transpose(0, 1)

        context = self.preprocess_layer(context)

        for i, layer in enumerate(self.layer_modules):

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                context = checkpoint(custom_layer(layer), context, mask_src)
            else:
                context = layer(context, mask_src)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': mask_src}

        return output_dict


class TransformerDecoder(nn.Module):
    """Decoder in 'Attention is all you need'"""

    def __init__(self, opt, embedding, positional_encoder, attribute_embeddings=None, ignore_source=False):
        """
        :param opt:
        :param embedding:
        :param positional_encoder:
        :param attribute_embeddings:
        :param ignore_source:
        """
        super(TransformerDecoder, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.transformer_hidden_dropout
        self.word_dropout = opt.transformer_word_dropout
        self.attn_dropout = opt.transformer_attn_dropout
        self.emb_dropout = opt.transformer_emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.encoder_type = opt.encoder_type
        self.ignore_source = ignore_source
        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.variational_dropout = opt.variational_dropout
        self.switchout = opt.switchout
        self.dec_gradient_checkpointing = getattr(opt, "dec_gradient_checkpointing", False) 

        if self.switchout > 0:
            self.transformer_word_dropout = 0

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        else:
            raise NotImplementedError

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.variational_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        # Using feature embeddings in models
        if attribute_embeddings is not None:
            self.use_feature = True
            self.attribute_embeddings = attribute_embeddings
            self.feature_projector = nn.Linear(opt.model_size + opt.model_size * attribute_embeddings.size(),
                                               opt.model_size)
        else:
            self.use_feature = None

        self.positional_encoder = positional_encoder

        if hasattr(self.positional_encoder, 'len_max'):
            len_max = self.positional_encoder.len_max
            mask = torch.ByteTensor(np.triu(np.ones((len_max, len_max)), k=1).astype('uint8'))
            self.register_buffer('mask', mask)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size,
                                                         self.dropout, self.inner_size,
                                                         self.attn_dropout, variational=self.variational_dropout,
                                                         ignore_source=self.ignore_source) for _ in range(self.layers)])

    def renew_buffer(self, new_len):

        #print(new_len)
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len+1, new_len+1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def process_embedding(self, input, atbs=None):

        input_ = input

        emb = embedded_dropout(self.word_lut, input_, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        if self.use_feature:
            len_tgt = emb.size(1)
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1).repeat(1, len_tgt, 1)  # B x H to 1 x B x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))
        return emb

    def create_custom_forward(self, module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    def forward(self, input, context, src, atbs=None, **kwargs):

        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        emb = self.process_embedding(input, atbs)

        if context is not None:
            mask_src = src.data.eq(onmt.Constants.SRC_PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.eq(onmt.Constants.TGT_PAD).byte().unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        # an ugly hack to bypass torch 1.2 breaking changes
        if torch_version >= 1.2:
            mask_tgt = mask_tgt.bool()

        output = self.preprocess_layer(emb.transpose(0, 1).contiguous())

        for i, layer in enumerate(self.layer_modules):
            if self.dec_gradient_checkpointing and self.training:
                output, coverage = checkpoint(self.create_custom_forward(layer), output, context, mask_tgt, mask_src)

            # if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
            #     output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)

            else:
                # coverage 作为attention, 其实返回后并没有在后续用到的样子
                output, coverage = layer(output, context, mask_tgt, mask_src)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}

        # return output, None
        return output_dict

    def step(self, input, decoder_state):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        atbs = decoder_state.tgt_atb
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq == True:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)

            src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1)
        else:
            input_ = input
        """ Embedding: batch_size x 1 x d_model """
        check = input_.gt(self.word_lut.num_embeddings)
        emb = self.word_lut(input_)

        """ Adding positional encoding """
        if self.time == 'positional_encoding':
            # print(emb.size())
            emb = emb * math.sqrt(self.model_size)
            emb = self.time_transformer(emb, t=input.size(1))
        else:
            # prev_h = buffer[0] if buffer is None else None
            # emb = self.time_transformer(emb, prev_h)
            # buffer[0] = emb[1]
            raise NotImplementedError

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        if self.use_feature:
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1)  # B x H to B x 1 x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if context is not None:
            if mask_src is None:
                if self.encoder_type == "audio":
                    if src.data.dim() == 3:
                        if self.encoder_cnn_downsampling:
                            long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.SRC_PAD)
                            mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                        else:
                            mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.SRC_PAD).unsqueeze(1)
                    elif self.encoder_cnn_downsampling:
                        long_mask = src.eq(onmt.Constants.SRC_PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.eq(onmt.Constants.SRC_PAD).unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.Constants.SRC_PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.eq(onmt.Constants.TGT_PAD).byte().unsqueeze(1)
        mask_tgt = mask_tgt + self.mask[:len_tgt, :len_tgt].type_as(mask_tgt)
        mask_tgt = torch.gt(mask_tgt, 0)
        # only get the final step of the mask during decoding (because the input of the network is only the last step)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        if torch_version >= 1.2:
            mask_tgt = mask_tgt.bool()

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)

            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage


class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None):
        super().__init__(encoder, decoder, generator)
        self.encoder = self.encoder
        if decoder.dec_pretrained_model == "transformer":  # no pretrained model for decoder
            self.model_size = self.decoder.model_size
            self.switchout = self.decoder.switchout
            self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)
        elif decoder.dec_pretrained_model == "roberta" or decoder.dec_pretrained_model == "bert":
            self.model_size = self.decoder.config.bert_hidden_size
            self.tgt_vocab_size = self.decoder.config.vocab_size
        elif decoder.dec_pretrained_model == "gpt2":
            self.model_size = self.decoder.config.n_embd
            self.tgt_vocab_size = self.decoder.config.vocab_size
        else:
            print("Warning: dec_pretrained_model is not correct")
            exit(-1)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, zero_encoder=False):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """

        src = batch.get('source')  # [src_len, b]
        tgt = batch.get('target_input')  # [tgt_len, b]
        tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src = src.transpose(0, 1)   # transpose to have batch first [b, src_len]
        tgt = tgt.transpose(0, 1)   # [b, tgt_len]
        src_attention_mask = src.ne(onmt.Constants.SRC_PAD).long()  # [b, src_len]

        segments_tensor = src.ne(onmt.Constants.SRC_PAD).long()

        if self.encoder.enc_pretrained_model == "transformer":
            if hasattr(self, 'pretrain_emb'):
                pretrain_emb_outputs = self.pretrain_emb(src, segments_tensor, src_attention_mask)
                embeddings = pretrain_emb_outputs[0]
                encoder_output = self.encoder(src, embeddings)
            else: 
                encoder_output = self.encoder(src)
            context = encoder_output['context']
            # [src_len, batch, d] => [batch, src_len, d]  # to make it consistent with bert
            context = context.transpose(0, 1)
        elif self.encoder.enc_pretrained_model == "bert" or self.encoder.enc_pretrained_model == "roberta":
            encoder_outputs = self.encoder(src, segments_tensor, src_attention_mask)  # the encoder is a pretrained model
            context = encoder_outputs[0]
        else:
            print("wrong enc_pretrained_model")
            exit(-1)

        # zero out the encoder part for pre-training
        if zero_encoder:
            context.zero_()

        if self.decoder.dec_pretrained_model == "transformer":
            # 在 decoder部分，我们用到了src 做mask_src 我不想改变这部分
            # src: [b, l]
            # context: [b, l, de_model]  =>  [l, b, de_model]
            context = context.transpose(0, 1)
            decoder_output = self.decoder(tgt, context, src, atbs=tgt_atb)
            output = decoder_output['hidden']

        elif self.decoder.dec_pretrained_model == "bert" or self.decoder.dec_pretrained_model == "roberta":
            # src: [b, l], src 用于做mask_src
            # context: [b, l, de_model]
            # tgt: 训练过程是tgt_input
            tgt_token_type = tgt.ne(onmt.Constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.ne(onmt.Constants.TGT_PAD).long()  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          token_type_ids=tgt_token_type,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          no_offset=True)
            decoder_output = decoder_output[0]
            decoder_output = decoder_output.transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            output = decoder_output
        elif self.decoder.dec_pretrained_model == "gpt2":
            tgt_attention_mask = tgt.ne(onmt.Constants.TGT_PAD).long()  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          token_type_ids=None,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          use_cache=False)
            decoder_output = decoder_output[0]
            decoder_output = decoder_output.transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            output = decoder_output

        else:
            print("Please check the dec_pretrained_model")
            raise NotImplementedError


        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output  # [tgt_len, bsz, d]
        output_dict['encoder'] = context  # [bsz, src_len, d]
        output_dict['src_mask'] = src_attention_mask  # [bsz, src_len]
        output_dict['target_mask'] = target_mask


        # This step removes the padding to reduce the load for the final layer
#        if target_masking is not None:
#            output = output.contiguous().view(-1, output.size(-1))  # not batch first
#
#            mask = target_masking  # not batch first
#            """ We remove all positions with PAD """
#            flattened_mask = mask.view(-1)
#
#            non_pad_indices = torch.nonzero(flattened_mask, as_tuple=False).squeeze(1)
#
#            output = output.index_select(0, non_pad_indices)

        # final layer: computing softmax
        logprobs = self.generator[0](output, log_softmax=False)
        output_dict['logprobs'] = logprobs

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_atb = batch.get('target_atb')  # a dictionary of attributes

        src = src.transpose(0, 1)  # [src_len, b] => [b, src_len]
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        src_attention_mask = src.ne(onmt.Constants.SRC_PAD).long()   #[b, src_len]
        segments_tensor = src.ne(onmt.Constants.SRC_PAD).long()  #[b, src_len]

        if self.encoder.enc_pretrained_model == "bert" or self.encoder.enc_pretrained_model == "roberta":
            encoder_outputs = self.encoder(src, segments_tensor, src_attention_mask)  # the encoder is a pretrained model
            # 在encoder里我们用 src 制作 src_mask，src保持和以前的代码不变
            context = encoder_outputs[0]
        elif self.encoder.enc_pretrained_model == "transformer" and hasattr(self, 'pretrain_emb'):
            pretrain_emb_outputs = self.pretrain_emb(src, segments_tensor, src_attention_mask)
            embeddings = pretrain_emb_outputs[0]
            encoder_output = self.encoder(src, embeddings)
            context = encoder_output['context']
            # [src_len, batch, d] => [batch, src_len, d]  # to make it consistent with bert
            context = context.transpose(0, 1)
        else:
            print("wrong enc_pretrained_model")
            exit(-1)

        # we don't have target, so check this part later

        if hasattr(self, 'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()
        decoder_output = self.decoder(tgt_input, context, src, atbs=tgt_atb)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder') and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):
            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_t)
            else:
                gen_t = self.generator(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.TGT_PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.TGT_PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, tgt_inputs, decoder_state):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        hidden, _ = self.decoder.step(tgt_inputs, decoder_state)

        # make time first
        if self.decoder.dec_pretrained_model == "bert" or self.decoder.dec_pretrained_model == "roberta":
            hidden = hidden.transpose(0, 1)
        elif self.decoder.dec_pretrained_model == "transformer":
            hidden = hidden
        else:
            raise NotImplementedError
        log_prob = self.generator[0](hidden.squeeze(0))

        output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        # output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1):
        """
        Generate a new decoder state based on the batch input
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')  # [src_len, bsz]
        tgt_atb = batch.get('target_atb')
        src_transposed = src.transpose(0, 1)  # make batch_size first (batch_size, src_len)
        segments_tensor = src_transposed.ne(onmt.Constants.SRC_PAD).long()
        src_attention_mask = src_transposed.ne(onmt.Constants.SRC_PAD).long()  #[batch_size, src_len]

        # by me
        if self.encoder.enc_pretrained_model == "transformer":
            if hasattr(self, 'pretrain_emb'):
                pretrain_emb_outputs = self.pretrain_emb(src, segments_tensor, src_attention_mask)
                embeddings = pretrain_emb_outputs[0]
                encoder_output = self.encoder(src_transposed, embeddings)
            else:
                encoder_output = self.encoder(src_transposed)
            #  [batch, src_len, d]  we need context not batch first
            context = encoder_output['context']
        elif self.encoder.enc_pretrained_model == "bert" or self.encoder.enc_pretrained_model == "roberta":
            encoder_outputs = self.encoder(src_transposed, segments_tensor, src_attention_mask) # the encoder is a pretrained model
            context = encoder_outputs[0]  # [batch_size , len, hidden]
            context = context.transpose(0, 1)  # [batch_size , len, hidden] => [len, batch_size, hidden]

        # src_transposed: batch first
        # [batch_size , len] => [batchsize, 1, len] 非padding位置True
        # src: time first
        dec_pretrained_model = self.decoder.dec_pretrained_model
        if dec_pretrained_model == "transformer":
            mask_src = src_transposed.eq(onmt.Constants.SRC_PAD).unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting
        elif dec_pretrained_model == "bert" or dec_pretrained_model == "roberta":
            mask_src = src_transposed.ne(onmt.Constants.SRC_PAD).unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting
        else:
            print("Warning: unknown dec_pretrained_model")
            raise NotImplementedError

        decoder_state = TransformerDecodingState(src, tgt_atb, context, mask_src,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, dec_pretrained_model=dec_pretrained_model)

        return decoder_state  


class TransformerDecodingState(DecoderState):

    def __init__(self, src, tgt_atb, context, src_mask, beam_size=1, model_size=512, type=1, dec_pretrained_model="transformer"):

        self.beam_size = beam_size
        self.model_size = model_size
        self.attention_buffers = dict()
        self.dec_pretrained_model = dec_pretrained_model

        if type == 1:
            # if audio only take one dimension since only used for mask
            self.original_src = src  # TxBxC
            self.concat_input_seq = True

            if src is not None:
                if src.dim() == 3:
                    self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
                    # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
                else:
                    self.src = src.repeat(1, beam_size)
            else:
                self.src = None

            if context is not None:
                self.context = context.repeat(1, beam_size, 1)
            else:
                self.context = None

            self.input_seq = None
            self.src_mask = None

            if tgt_atb is not None:
                self.use_attribute = True
                self.tgt_atb = tgt_atb
                # self.tgt_atb = tgt_atb.repeat(beam_size)  # size: Bxb
                for i in self.tgt_atb:
                    self.tgt_atb[i] = self.tgt_atb[i].repeat(beam_size)
            else:
                self.tgt_atb = None

        elif type == 2:
            bsz = context.size(1)
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(context.device)
            self.context = context.index_select(1, new_order)
            self.src = src.index_select(1, new_order)  # because src is batch first
            self.src_mask = src_mask.index_select(0, new_order)
            self.concat_input_seq = False

            if tgt_atb is not None:
                self.use_attribute = True
                self.tgt_atb = tgt_atb
                # self.tgt_atb = tgt_atb.repeat(beam_size)  # size: Bxb
                for i in self.tgt_atb:
                    self.tgt_atb[i] = self.tgt_atb[i].index_select(0, new_order)
            else:
                self.tgt_atb = None

        else:
            raise NotImplementedError


    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer  # dict of 2 keys (k, v) : T x B x H

    def update_beam(self, beam, b, remaining_sents, idx):

        if self.beam_size == 1:
            return

        for tensor in [self.src, self.input_seq]:

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        if self.tgt_atb is not None:
            for i in self.tgt_atb:
                tensor = self.tgt_atb[i]

                state_ = tensor.view(self.beam_size, remaining_sents)[:, idx]

                state_.copy_(state_.index_select(0, beam[b].getCurrentOrigin()))

                self.tgt_atb[i] = tensor

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            if buffer_ is None:
                continue

            for k in buffer_:
                t_, br_, d_ = buffer_[k].size()
                sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                sent_states.data.copy_(sent_states.data.index_select(
                    1, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active_with_hidden(t):
            if t is None:
                return t
            dim = t.size(-1)
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_without_hidden(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active_with_hidden(self.context)

        self.input_seq = update_active_without_hidden(self.input_seq)

        if self.src.dim() == 2:
            self.src = update_active_without_hidden(self.src)
        elif self.src.dim() == 3:
            t = self.src
            dim = t.size(-1)
            view = t.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            self.src = new_t

        if self.tgt_atb is not None:
            for i in self.tgt_atb:
                self.tgt_atb[i] = update_active_without_hidden(self.tgt_atb[i])

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active_with_hidden(buffer_[k])

    # For the new decoder version only
    def _reorder_incremental_state(self, reorder_state):
        self.context = self.context.index_select(1, reorder_state)

        self.src_mask = self.src_mask.index_select(0, reorder_state)

        if self.tgt_atb is not None:
            for i in self.tgt_atb:
                self.tgt_atb[i] = self.tgt_atb[i].index_select(0, reorder_state)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    if self.dec_pretrained_model == "transformer":
                        buffer_[k] = buffer_[k].index_select(1, reorder_state)  # 1 for time first
                    elif self.dec_pretrained_model == "bert" or  self.dec_pretrained_model == "roberta": 
                        buffer_[k] = buffer_[k].index_select(0, reorder_state)  # 0 for batch first
                    else:
                        raise NotImplementedError 


