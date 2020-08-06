from pytorch_pretrained_bert import BertTokenizer, BertModel
import onmt.Markdown
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-test_tgt', required=True,
                    help="Path to the validation target data")
opt = parser.parse_args()


tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')


def make_vecs(file_raw, file_tok, lang):
    if lang == "en":
        tokenizer = tokenizer_en
    elif lang == "zh":
        tokenizer = tokenizer_zh

    with open(file_raw, "r", encoding="utf-8")  as gf:
        segments_ids = []
        indexed_tokens = []
        marked_sents = []
        tokenized_sents = []
        for line in gf:
            sent = line.strip()
            marked_sent = "[CLS] " + sent + " [SEP]"
            tokenized_sent = tokenizer.tokenize(marked_sent)
            indexed_token = tokenizer.convert_tokens_to_ids(tokenized_sent)
            segments_id = [1] * len(tokenized_sent)

            # 每一句话作为一个元素添加进来
            marked_sents.append(marked_sent)
            tokenized_sents.append(tokenized_sent)
            indexed_tokens.append(indexed_token)
            segments_ids.append(segments_id)

    with open(file_tok, "w", encoding="utf-8")  as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')

    # number of sentences
    sentences_num = len(indexed_tokens)
    print("number of sentences:", sentences_num)

    # 最长的句子的长度
    maxlen = max([len(indexed_tokens[i]) for i in range(sentences_num)])
    print("the max length of the sentences:", maxlen)




def main():
    print("process valid en start")
    valid_en_raw = "/project/student_projects2/dhe/Bert/data/ted.valid.en"
    valid_en_tok = "/project/student_projects2/dhe/Bert/data/ted.valid.en.tok.bert"
    # bert_en_vecs = make_vecs(file_en_raw, file_en_tok, lang="en")
    make_vecs(valid_en_raw, valid_en_tok, lang="en")
    print("process valid en end")

    print("process valid zh start")
    valid_zh_raw = "/project/student_projects2/dhe/Bert/data/ted.valid.zh.char"
    valid_zh_tok = "/project/student_projects2/dhe/Bert/data/ted.valid.zh.tok.bert"
    # bert_zh_vecs = make_vecs(file_zh_raw, file_zh_tok, lang="zh")
    make_vecs(valid_zh_raw, valid_zh_tok, lang="zh")
    print("process valid zh end")



    print("process train en start")
    train_en_raw = "/project/student_projects2/dhe/Bert/data/ted.train.en"
    train_en_tok = "/project/student_projects2/dhe/Bert/data/ted.train.en.tok.bert"
    # bert_zh_vecs = make_vecs(file_zh_raw, file_zh_tok, lang="zh")
    make_vecs(train_en_raw, train_en_tok, lang="zh")
    print("process train en end")


    print("process train zh start")
    train_zh_raw = "/project/student_projects2/dhe/Bert/data/ted.train.zh.char"
    train_zh_tok = "/project/student_projects2/dhe/Bert/data/ted.train.zh.tok.bert"
    # bert_zh_vecs = make_vecs(file_zh_raw, file_zh_tok, lang="zh")
    make_vecs(train_zh_raw, train_zh_tok, lang="zh")
    print("process train zh end")

    # return bert_en_vecs, bert_zh_vecs


if __name__ == "__main__":
    main()









