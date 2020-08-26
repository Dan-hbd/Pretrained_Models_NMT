from __future__ import division
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import time
import datetime
from onmt.train_utils.trainer import XETrainer
from onmt.modules.Loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.ModelConstructor import build_model
from options import make_parser
from collections import defaultdict

print("cuda is available: ", torch.cuda.is_available())

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()
print(opt)

# An ugly hack to have weight norm on / off
onmt.Constants.weight_norm = opt.weight_norm
onmt.Constants.checkpointing = opt.checkpointing
onmt.Constants.max_position_length = opt.max_position_length

# Use static dropout if checkpointing > 0
if opt.checkpointing > 0:
    onmt.Constants.static = True
if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

torch.manual_seed(opt.seed)


def main():

    if opt.data_format == 'raw':
        start = time.time()

        if opt.data.endswith(".train.pt"):
            print("Loading data from '%s'" % opt.data)
            dataset = torch.load(opt.data)
        else:
            print("Loading data from %s" % opt.data + ".train.pt")
            dataset = torch.load(opt.data + ".train.pt")

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

        # For backward compatibility
        train_dict = defaultdict(lambda: None, dataset['train'])
        valid_dict = defaultdict(lambda: None, dataset['valid'])

        train_data = onmt.Dataset(train_dict['src'], train_dict['tgt'],
                                  train_dict['src_atbs'], train_dict['tgt_atbs'],
                                  batch_size_words=opt.batch_size_words,
                                  data_type=dataset.get("type", "text"),
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier,
                                  augment=opt.augment_speech,
                                  upsampling=opt.upsampling)
        valid_data = onmt.Dataset(valid_dict['src'], valid_dict['tgt'],
                                  valid_dict['src_atbs'], valid_dict['tgt_atbs'],
                                  batch_size_words=opt.batch_size_words,
                                  data_type=dataset.get("type", "text"),
                                  batch_size_sents=opt.batch_size_sents,
                                  upsampling=opt.upsampling)

        dicts = dataset['dicts']

        print(' * number of training sentences. %d' % len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    elif opt.data_format == 'bin':
        print("Loading memory binned data files ....")
        start = time.time()
        from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

        dicts = torch.load(opt.data + ".dict.pt")

        train_path = opt.data + '.train'
        train_src = IndexedInMemoryDataset(train_path + '.src')
        train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

        train_data = onmt.Dataset(train_src,
                                  train_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = IndexedInMemoryDataset(valid_path + '.src')
        valid_tgt = IndexedInMemoryDataset(valid_path + '.tgt')

        valid_data = onmt.Dataset(valid_src,
                                  valid_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents)

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)
    elif opt.data_format == 'mmem':
        print("Loading memory mapped data files ....")
        start = time.time()
        from onmt.data_utils.MMapIndexedDataset import MMapIndexedDataset

        dicts = torch.load(opt.data + ".dict.pt")

        train_path = opt.data + '.train'
        train_src = MMapIndexedDataset(train_path + '.src')
        train_tgt = MMapIndexedDataset(train_path + '.tgt')

        train_data = onmt.Dataset(train_src,
                                  train_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = MMapIndexedDataset(valid_path + '.src')
        valid_tgt = MMapIndexedDataset(valid_path + '.tgt')

        valid_data = onmt.Dataset(valid_src,
                                  valid_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents)
        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

    else:
        raise NotImplementedError

    additional_data = []
    if opt.additional_data != "none":
        add_data = opt.additional_data.split(";")
        add_format = opt.additional_data_format.split(";")
        assert(len(add_data) == len(add_format))
        for i in range(len(add_data)):
            if add_format[i] == 'raw':
                if add_data[i].endswith(".train.pt"):
                    print("Loading data from '%s'" % opt.data)
                    add_dataset = torch.load(add_data[i])
                else:
                    print("Loading data from %s" % opt.data + ".train.pt")
                    add_dataset = torch.load(add_data[i] + ".train.pt")

                additional_data.append(onmt.Dataset(add_dataset['train']['src'],
                                          dataset['train']['tgt'], batch_size_words=opt.batch_size_words,
                                          data_type=dataset.get("type", "text"),
                                          batch_size_sents=opt.batch_size_sents,
                                          multiplier=opt.batch_size_multiplier,
                                          reshape_speech=opt.reshape_speech,
                                          augment=opt.augment_speech))
            elif add_format[i] == 'bin':

                from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

                train_path = add_data[i] + '.train'
                train_src = IndexedInMemoryDataset(train_path + '.src')
                train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

                additional_data.append(onmt.Dataset(train_src,
                                       train_tgt,
                                       batch_size_words=opt.batch_size_words,
                                       data_type=opt.encoder_type,
                                       batch_size_sents=opt.batch_size_sents,
                                       multiplier = opt.batch_size_multiplier))

    if opt.load_from:
        checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
        print("* Loading dictionaries from the checkpoint")
        dicts = checkpoint['dicts']
    else:
        dicts['tgt'].patch(opt.patch_vocab_multiplier)
        checkpoint = None

    if "src" in dicts:
        print(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
    else:
        print(' * vocabulary size. target = %d' %
              (dicts['tgt'].size()))

    print('Building model...')

    model = build_model(opt, dicts)

    if opt.get_context_emb:
        assert opt.enc_pretrained_model == "transformer"
        assert opt.dec_pretrained_model == "transformer"
        print("We use pretrained model to get contextualized embeddings and feed them to the original transformer")

        if opt.get_context_emb == "bert":
            from pretrain_module.configuration_bert import BertConfig
            from pretrain_module.modeling_bert import BertModel
            emb_pretrain_config = BertConfig.from_json_file(opt.emb_pretrained_config_dir + "/" + opt.emb_config_name)
            pretrain_emb = BertModel(emb_pretrain_config,
                                     bert_word_dropout=opt.emb_pretrain_word_dropout,
                                     bert_emb_dropout=opt.emb_pretrain_emb_dropout,
                                     bert_atten_dropout=opt.emb_pretrain_attn_dropout,
                                     bert_hidden_dropout=opt.emb_pretrain_hidden_dropout,
                                     bert_hidden_size=opt.emb_pretrain_hidden_size,
                                     is_decoder=False,
                                     )
            emb_state_dict_file = opt.emb_pretrained_config_dir + "/" + opt.emb_pretrained_state_dict
            emb_model_state_dict = torch.load(emb_state_dict_file, map_location="cpu")
            print("After builing pretrained model we load the state from:\n", emb_state_dict_file)
            pretrain_emb.from_pretrained(pretrained_model_name_or_path=opt.enc_pretrained_config_dir,
                                         model=pretrain_emb,
                                         output_loading_info=True,
                                         state_dict=emb_model_state_dict,
                                         model_prefix=opt.get_context_emb
                                         )
            model.add_module("pretrain_emb", pretrain_emb)
        else:
            print("Warning: contextualized embeddings can be only got from bert or roberta")
            exit(-1)

    """ Building the loss function """
    loss_function = NMTLossFunc(dicts['tgt'].size(),
                                label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of all parameters: %d' % n_params)

    n_params_grad = sum([p.nelement() for p in model.parameters() if p.requires_grad==True])
    print('* number of all parameters that need gradient: %d' % n_params_grad)

    n_params_nograd = sum([p.nelement() for p in model.parameters() if p.requires_grad==False])
    print('* number of all parameters that do not need gradient: %d' % n_params_nograd)

    assert n_params == (n_params_grad + n_params_nograd)
    print(model)

    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        raise NotImplementedError("Warning! Multi-GPU training is not fully tested and potential bugs can happen.")
    else:
        trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True)
        if len(additional_data) > 0:
            trainer.add_additional_data(additional_data, opt.data_ratio)

    trainer.run(checkpoint=checkpoint)


if __name__ == "__main__":
    main()
