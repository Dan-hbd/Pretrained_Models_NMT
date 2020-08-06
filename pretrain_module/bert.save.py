import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# 这句话只会控制打印出来的小数位数，不会数据本身的精度
torch.set_printoptions(precision=4)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')


def create_tokenized_data(file_raw):

    with open(file_raw,"r",encoding="utf-8")  as f_raw:
        tokenized_sents = []
        marked_sents = []
        indexed_tokens = []
        segments_ids =[]
        # line_num =0
        for line in f_raw:
            sent = line.strip()
            marked_sent ="[CLS] " + sent + " [SEP]"
            tokenized_sent =tokenizer.tokenize(marked_sent)

            indexed_token = tokenizer.convert_tokens_to_ids(tokenized_sent)
            segments_id = [1] * len(tokenized_sent)

            # 每一句话作为一个元素添加进来
            marked_sents.append(marked_sent)
            tokenized_sents.append(tokenized_sent)
            indexed_tokens.append(indexed_token)
            segments_ids.append(segments_id)



            # if line_num == 0:
            #     print("line_num:", line_num, tokenized_sents[0])
            #     tokenized_sents_0 = tokenized_sents[0]
            # if line_num == 1:
            #     print("line_num:", line_num,tokenized_sents[1])
            #
            # line_num += 1

    # number of sentences
    sentences_num = len(indexed_tokens)

    # 最长的句子的长度
    maxlen = max([len(indexed_tokens[i]) for i in range (sentences_num)])

    # 把所有的句子都padding到这个长度，padding的值为0
    for i in range(sentences_num):
        indexed_tokens[i].extend([0]*(maxlen-len(indexed_tokens[i])))
        segments_ids[i].extend([0]*(maxlen-len(segments_ids[i])))

    # 把 indexed_tokens  segments_ids 转换为tensor

    tokens_tensor = torch.tensor(indexed_tokens)  #【batch_size, sent_len】 [1,7]
    segments_tensor = torch.tensor(segments_ids) #【batch_size, sent_len】 padding 位置value 为0， 其他为1


    model.eval()


    # Predict hidden states features for each layer, no backward, so no gradient
    with torch.no_grad():
        # encoded_layers is a list, 12 layers in total, for every element of the list : 【batch_size, sent_len, hidden_size】
        encoded_layers, _ = model(tokens_tensor, segments_tensor)
        # combine 12 layers to make this one whole big Tensor
        token_embeddings = torch.stack(encoded_layers, dim=0)

    return token_embeddings


def main():
    file1= "/project/student_projects2/dhe/Bert/data/data.1"
    file2= "/project/student_projects2/dhe/Bert/data/data.100"
    tensor1 = create_tokenized_data(file1)  # [12, 1,9,768]
    tensor1_new =torch.squeeze(tensor1, dim=1)   # [12,9,768]

    tensor2 = create_tokenized_data(file2)  # [12,2, 9, 768]

    lst2 = [tensor2[i][0] for i in range(12)]
    tensor2_new = torch.stack(lst2 ,0)

    print("let's compare")
    # [batch_size, sent_length, hidden], dim=2 比较的是hidden 的维度的向量
    similarity = torch.nn.functional.cosine_similarity(tensor1_new,tensor2_new,dim=2)
    print("the similarities:",similarity)
    # print(torch.equal(tensor1_new, tensor2_new))


    # file1= "/project/student_projects2/dhe/Bert/data/data.1"
    # file2= "/project/student_projects2/dhe/Bert/data/data.100"
    # list1 = create_tokenized_data(file1)  # [12, 1,9,768]
    #
    # list2 = create_tokenized_data(file2)  # [12,2, 9, 768]
    # count = 0
    # for layer in range(len(list1)):
    #     print(torch.nn.functional.cosine_similarity(list1,list2[0],dim=2))





    # print("let's compare")
    # print(torch.equal(tensor1_new, tensor2_new))



if __name__ == "__main__":
    main()


