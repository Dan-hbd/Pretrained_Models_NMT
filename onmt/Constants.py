

# for Bert, both en and zh, and for roberta_zh
BERT_PAD = 0
BERT_UNK = 100
BERT_BOS = 101
BERT_EOS = 102

## 数据处理不会用到以前Chinese Project的字典
#TRANSFORMER_PAD = 0
#TRANSFORMER_UNK = 1
#TRANSFORMER_BOS = 2
#TRANSFORMER_EOS = 3


# for Roberta_en
EN_ROBERTA_PAD = 1
EN_ROBERTA_UNK = 3
EN_ROBERTA_BOS = 0
EN_ROBERTA_EOS = 2

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
