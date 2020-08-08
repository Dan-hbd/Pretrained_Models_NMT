import sys
sys.path.append('/home/dhe/hiwi/Exercises/Pretrained_Models_NMT/')

from pytorch_pretrained_bert import BertTokenizer
import onmt.Markdown
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-src_data', required=True,
                    help="Path to the source data")
parser.add_argument('-tgt_data', required=True,
                    help="Path to the target data")

opt = parser.parse_args()


def tokenize_data(raw_data, tokenizer, lang):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            # 我这里特意用marked_sent 是为了说明，不管是src还是tgt我没有加CLS SEP
            if lang == "en":
            # 虽然我们把字典里的"[CLS] 换成了<s>，但是因为我们这里用的tokenizer,还是要用原来的特殊字符,否则tokenize会出问题"
                #marked_sent = "<s> " + sent + " </s>"
                marked_sent = "[CLS] " + sent + " [SEP]"
            elif lang == "zh":
                marked_sent = sent
            tokenized_sent = tokenizer.tokenize(marked_sent)
            tokenized_sents.append(tokenized_sent)

    new_data = raw_data + ".bert.tok"
    with open(new_data, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')


def main():
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

    src_lang = "en"
    tgt_lang = "zh"

    tokenize_data(opt.src_data, tokenizer_en, src_lang)

    tokenize_data(opt.tgt_data, tokenizer_zh, tgt_lang)


if __name__ == "__main__":
    main()
