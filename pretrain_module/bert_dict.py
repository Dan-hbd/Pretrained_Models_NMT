import sys
sys.path.append('/home/dhe/hiwi/Exercises/Pretrained_Models_NMT/')
import onmt.Markdown
import argparse


parser = argparse.ArgumentParser(description='bert_dict.py')

onmt.Markdown.add_md_help_argument(parser)
parser.add_argument('-bert_src_vocab', required=True,
                    help="Path to the bert_src_vocab data")
parser.add_argument('-src_lang', required=True,
                    help="src language")

parser.add_argument('-bert_tgt_vocab', required=True,
                    help="Path to the bert_tgt_vocab data")
parser.add_argument('-tgt_lang', required=True,
                    help="tgt language")


opt = parser.parse_args()


def load_vocab(vocab_file, lang):
    """Loads a vocabulary file into a dictionary."""
    index = 0
    vocab = open(vocab_file, "r")
    word2idx = open("bert_word2idx."+lang, "w")
    idx2word = open("bert_idx2word."+lang, "w")
    while True:
        word = vocab.readline()
        # 读到最后一行
        if not word:
            break
        # 去掉word 后面原本的换行符
        word = word.strip()
        # 写入文件在行尾加上换行符
        idx2word.write(str(index) + " " + word + "\n")
        word2idx.write(word + " " + str(index) + "\n")
        index += 1

    vocab.close()
    word2idx.close()
    idx2word.close()


load_vocab(opt.bert_src_vocab, opt.src_lang)
load_vocab(opt.bert_tgt_vocab, opt.tgt_lang)






