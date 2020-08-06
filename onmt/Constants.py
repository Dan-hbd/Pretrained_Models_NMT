
# for our transformer
SRC_PAD = 0
SRC_UNK = 1
SRC_BOS = 2
SRC_EOS = 3

TGT_PAD = 0
TGT_UNK = 100
TGT_BOS = 101
TGT_EOS = 102



# for Bert
BERT_PAD = 0
BERT_UNK = 100
BERT_BOS = 101
BERT_EOS = 102


# for Roberta
EN_ROBERTA_PAD = 1
EN_ROBERTA_UNK = 3
EN_ROBERTA_BOS = 0
EN_ROBERTA_EOS = 2


ZH_ROBERTA_PAD = 0
ZH_ROBERTA_UNK = 100
ZH_ROBERTA_BOS = 101
ZH_ROBERTA_EOS = 102


SRC_PAD_WORD = '[PAD]'
SRC_UNK_WORD = '[UNK]'
SRC_BOS_WORD = '<s>'
SRC_EOS_WORD = '</s>'

TGT_PAD_WORD = '[PAD]'
TGT_UNK_WORD = '[UNK]'
TGT_BOS_WORD = '<s>'
TGT_EOS_WORD = '</s>'


checkpointing = 0
static = False
residual_type = 'regular'
max_position_length = 8192

