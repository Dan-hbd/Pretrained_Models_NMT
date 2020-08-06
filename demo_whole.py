import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


text1 = "I love eating apple very much"
marked_text1 = "[CLS] " + text1 + " [SEP]"
tokenized_text1 = tokenizer.tokenize(marked_text1)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
segments_ids1 = [1] * len(tokenized_text1)

tokens_tensor1 =  torch.tensor([indexed_tokens1])
segments_tensors1 = torch.tensor([segments_ids1])


text2 = "I love eating apple very much" \
        "not the iphone from apple company"

marked_text2 = "[CLS] " + text2 + " [SEP]"
tokenized_text2 = tokenizer.tokenize(marked_text2)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
segments_ids2 = [1] * len(tokenized_text2)

tokens_tensor2 =  torch.tensor([indexed_tokens2])
segments_tensors2 = torch.tensor([segments_ids2])


indexed_tokens3 = [indexed_tokens1, indexed_tokens2]
segments_ids3 = [segments_ids1, segments_ids2]
tokens_tensor3 =  torch.tensor([indexed_tokens3])
segments_tensors3 = torch.tensor([segments_ids3])


model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
