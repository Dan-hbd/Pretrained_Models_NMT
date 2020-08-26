import torch
import torch.nn as nn
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding

init = torch.nn.init

MAX_LEN = onmt.Constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):

    model = None

    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'

    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'

    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'

    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'

    if not hasattr(opt, 'input_size'):
        opt.input_size = 40

    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'xavier'

    if not hasattr(opt, 'ctc_loss'):
        opt.ctc_loss = 0

    if not hasattr(opt, 'encoder_layers'):
        opt.encoder_layers = -1

    if not hasattr(opt, 'fusion'):
        opt.fusion = False

    if not hasattr(opt, 'cnn_downsampling'):
        opt.cnn_downsampling = False

    if not hasattr(opt, 'switchout'):
        opt.switchout = 0.0

    if not hasattr(opt, 'variational_dropout'):
        opt.variational_dropout = False

    if not hasattr(opt, 'enc_ln_before'):
        opt.enc_ln_before = False
    if not hasattr(opt, 'get_context_emb'):
        opt.get_context_emb = ""


    if opt.enc_pretrained_model == 'bert':
        onmt.Constants.SRC_PAD = onmt.Constants.BERT_PAD
        onmt.Constants.SRC_UNK = onmt.Constants.BERT_UNK
        onmt.Constants.SRC_BOS = onmt.Constants.BERT_BOS
        onmt.Constants.SRC_EOS = onmt.Constants.BERT_EOS
    elif opt.enc_pretrained_model == 'roberta':
        onmt.Constants.SRC_PAD = onmt.Constants.EN_ROBERTA_PAD
        onmt.Constants.SRC_UNK = onmt.Constants.EN_ROBERTA_UNK
        onmt.Constants.SRC_BOS = onmt.Constants.EN_ROBERTA_BOS
        onmt.Constants.SRC_EOS = onmt.Constants.EN_ROBERTA_EOS
    elif opt.enc_pretrained_model == 'transformer':
        onmt.Constants.SRC_PAD = onmt.Constants.TRANSFORMER_PAD
        onmt.Constants.SRC_UNK = onmt.Constants.TRANSFORMER_UNK
        onmt.Constants.SRC_BOS = onmt.Constants.TRANSFORMER_BOS
        onmt.Constants.SRC_EOS = onmt.Constants.TRANSFORMER_EOS
    else:
        print("Warning: wrong enc_pretrained_model")
        exit(-1)
    
    # 不管哪种structure的decoder,处理tgt数据要么用的bert 要么roberta的字典，中文的这两个预训练模型有相同的字典
    if opt.dec_pretrained_model == 'bert' or opt.dec_pretrained_model == 'roberta' or opt.dec_pretrained_model == 'transformer':
        onmt.Constants.TGT_PAD = onmt.Constants.BERT_PAD
        onmt.Constants.TGT_UNK = onmt.Constants.BERT_UNK
        onmt.Constants.TGT_BOS = onmt.Constants.BERT_BOS
        onmt.Constants.TGT_EOS = onmt.Constants.BERT_EOS
    else:
        print("Warning: wrong dec_pretrained_model")
        exit(-1)



    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # BUILD GENERATOR
    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    if 'src' in dicts and opt.get_context_emb == "" and opt.enc_pretrained_model == "transformer":
        embedding_src = nn.Embedding(dicts['src'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.SRC_PAD)
    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    elif opt.dec_pretrained_model == "transformer":
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.TGT_PAD)
    else:
        print("we build a pretrained model for decoder, no embedding_tgt is needed")

    if opt.model == 'transformer':
        onmt.Constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            print("\n")
            print("Building Encoder start")
            if opt.enc_pretrained_model == "transformer":
                opt.enc_not_load_state = True
                print("Encoder is not initialized from pretrained model")
                encoder = TransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_type)

            else:
                print("Build a pretrained model: {}, for encoder".format(opt.enc_pretrained_model))
                if opt.enc_pretrained_model == "bert":
                    from pretrain_module.configuration_bert import BertConfig
                    from pretrain_module.modeling_bert import BertModel

                    enc_bert_config = BertConfig.from_json_file(opt.enc_pretrained_config_dir + "/" + opt.enc_config_name)
                    encoder = BertModel(enc_bert_config,
                                                    bert_word_dropout=opt.enc_pretrain_word_dropout,
                                                    bert_emb_dropout=opt.enc_pretrain_emb_dropout,
                                                    bert_atten_dropout=opt.enc_pretrain_attn_dropout,
                                                    bert_hidden_dropout=opt.enc_pretrain_hidden_dropout,
                                                    bert_hidden_size=opt.enc_pretrain_hidden_size,
                                                    is_decoder=False,
                                               )
                elif opt.enc_pretrained_model == "roberta":
                    from pretrain_module.configuration_roberta import RobertaConfig
                    from pretrain_module.modeling_roberta import RobertaModel

                    enc_roberta_config = RobertaConfig.from_json_file(opt.enc_pretrained_config_dir + "/" + opt.enc_config_name)
                    encoder = RobertaModel(enc_roberta_config,
                                           bert_word_dropout=opt.enc_pretrain_word_dropout,
                                           bert_emb_dropout=opt.enc_pretrain_emb_dropout,
                                           bert_atten_dropout=opt.enc_pretrain_attn_dropout,
                                           bert_hidden_dropout=opt.enc_pretrain_hidden_dropout,
                                           bert_hidden_size=opt.enc_pretrain_hidden_size,
                                           is_decoder=False,
                                           encoder_normalize_before=opt.enc_ln_before,
                                         )
                    print("enc_ln_beforei:",opt.enc_ln_before,)
                else:
                    print("Warning: now only bert and roberta pretrained models are implemented:")
                    exit(-1)

                print("----------------opt.enc_not_load_state:", opt.enc_not_load_state)
                if opt.enc_not_load_state:
                    print("We do not load the state from pytorch")
                else:
                    enc_state_dict_file=opt.enc_pretrained_config_dir + "/" + opt.enc_state_dict
                    print("Loading weights from pretrained:\n",enc_state_dict_file)

                    enc_model_state_dict = torch.load(enc_state_dict_file, map_location="cpu")

                    encoder.from_pretrained(pretrained_model_name_or_path=opt.enc_pretrained_config_dir,
                                            model=encoder,
                                            output_loading_info=True,
                                            state_dict=enc_model_state_dict,
                                            model_prefix=opt.enc_pretrained_model
                                            )

            encoder.enc_pretrained_model = opt.enc_pretrained_model

        print("Building Decoder start")
        if opt.dec_pretrained_model == "transformer":
            print("Decoder is not initialized from pretrained model")
            decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, attribute_embeddings=None)
        else:
            print("Build a pretrained model: {}, for decoder".format(opt.dec_pretrained_model))
            if opt.dec_pretrained_model == "bert":
                if opt.enc_pretrained_model != "bert":
                    from pretrain_module.configuration_bert import BertConfig
                    from pretrain_module.modeling_bert import BertModel
                dec_bert_config = BertConfig.from_json_file(opt.dec_pretrained_config_dir + "/" + opt.dec_config_name)
                decoder = BertModel(dec_bert_config,
                                    bert_word_dropout=opt.dec_pretrain_word_dropout,
                                    bert_emb_dropout=opt.dec_pretrain_emb_dropout,
                                    bert_atten_dropout=opt.dec_pretrain_attn_dropout,
                                    bert_hidden_dropout=opt.dec_pretrain_hidden_dropout,
                                    bert_hidden_size=opt.dec_pretrain_hidden_size,
                                    is_decoder=True,
                                    )

            elif opt.dec_pretrained_model == "roberta":
                print("Pretrained model {} is applied to decoder".format(opt.dec_pretrained_model))
                if opt.enc_pretrained_model != "roberta":
                    from pretrain_module.configuration_roberta import RobertaConfig
                    from pretrain_module.modeling_roberta import RobertaModel

                dec_roberta_config = RobertaConfig.from_json_file(opt.dec_pretrained_config_dir + "/" + opt.dec_config_name)
                decoder = RobertaModel(dec_roberta_config,
                                        bert_word_dropout=opt.dec_pretrain_word_dropout,
                                        bert_emb_dropout=opt.dec_pretrain_emb_dropout,
                                        bert_atten_dropout=opt.dec_pretrain_attn_dropout,
                                        bert_hidden_dropout=opt.dec_pretrain_hidden_dropout,
                                        bert_hidden_size=opt.dec_pretrain_hidden_size,
                                        is_decoder=True,
                                     )
            else:
                print("Warning: now only bert and roberta pretrained models are implemented:")
                exit(-1)
            
            if opt.dec_not_load_state:
                print("We don't load the state for pretrained model of decoder from pytorch")

            else:
                dec_state_dict_file=opt.dec_pretrained_config_dir + "/" + opt.dec_state_dict
                print("After builing pretrained model we load the state from:\n",dec_state_dict_file)
                print("The pretrained model is:",opt.dec_pretrained_model)

                dec_model_state_dict = torch.load(dec_state_dict_file, map_location="cpu")

                decoder.from_pretrained(pretrained_model_name_or_path=opt.dec_pretrained_config_dir,
                                        model=decoder,
                                        output_loading_info=True,
                                        state_dict=dec_model_state_dict,
                                        model_prefix=opt.dec_pretrained_model
                                        )

        decoder.dec_pretrained_model = opt.dec_pretrained_model
        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    else:
        raise NotImplementedError

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)


    if opt.dec_pretrained_model != "transformer" or opt.enc_pretrained_model != "transformer":
        opt.init_embedding =""

    if opt.init_embedding == 'xavier':
        if model.encoder.word_lut is not None:
            init.xavier_uniform_(model.encoder.word_lut.weight)
        if model.decoder.word_lut is not None:
            init.xavier_uniform_(model.decoder.word_lut.weight)

    elif opt.init_embedding == 'normal':
        if model.encoder.word_lut is not None:
            init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
        if model.decoder.word_lut is not None:
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def init_model_parameters(model, opt):

    # currently this function does not do anything
    # because the parameters are locally initialized
    pass


def build_language_model(opt, dicts):

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    from onmt.modules.LSTMLM.Models import LSTMLMDecoder, LSTMLM

    decoder = LSTMLMDecoder(opt, dicts['tgt'])

    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    model = LSTMLM(None, decoder, nn.ModuleList(generators))

    if opt.tie_weights:
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def build_fusion(opt, dicts):

    # the fusion model requires a pretrained language model
    print("Loading pre-trained language model from %s" % opt.lm_checkpoint)
    lm_checkpoint = torch.load(opt.lm_checkpoint, map_location=lambda storage, loc: storage)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_tm_model(opt, dicts)

    from onmt.modules.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model
