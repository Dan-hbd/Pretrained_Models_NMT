import sys
sys.path.append('/home/dhe/hiwi/Exercises/Init_encoder_with_Roberta/')
import onmt.Markdown
import argparse


parser = argparse.ArgumentParser(description='roberta_dict.py')

onmt.Markdown.add_md_help_argument(parser)
parser.add_argument('-roberta_src_vocab', required=True,
                    help="Path to the roberta_src_vocab data")
parser.add_argument('-src_lang', required=True,
                    help="src language")

parser.add_argument('-roberta_tgt_vocab', required=True,
                    help="Path to the roberta_tgt_vocab data")
parser.add_argument('-tgt_lang', required=True,
                    help="tgt language")


opt = parser.parse_args()


def load_vocab(vocab_file, lang):
    """Loads a vocabulary file into a dictionary."""
    index = 0
    vocab = open(vocab_file, "r")
    word2idx = open("roberta_word2idx."+lang, "w")
    idx2word = open("roberta_idx2word."+lang, "w")
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


load_vocab(opt.roberta_src_vocab, opt.src_lang)
load_vocab(opt.roberta_tgt_vocab, opt.tgt_lang)






