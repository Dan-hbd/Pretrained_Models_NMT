import torch
import argparse


parser = argparse.ArgumentParser(description='extract_finetuned_bert.py')
parser.add_argument('-model_dir', default='', type=str,
                    help='Path to pretrained model file.')
parser.add_argument('-save_dir', default='', type=str,
                    help='Path to extracted model from pretrained model ')

parser.add_argument('-finetuned_model', default='', type=str,
                    help='name of pretrained model ')
parser.add_argument('-extracted_model', default='', type=str,
                    help='name of the extracted model from pretrained model ')
def main():
    opt = parser.parse_args()
    model_dir = opt.model_dir
    save_dir = opt.save_dir
    fintuned_model = opt.finetuned_model
    extracted_model = opt.extracted_model
    
    input_model = opt.model_dir + opt.finetuned_model
    checkpoint = torch.load(input_model)
    output = opt.save_dir + opt.extracted_model

    print("extract the  model from :",input_model)
    print("before deleting, the keys are:", checkpoint.keys())
    for key_item in ['optim', 'dicts', 'opt', 'epoch', 'iteration', 'batchOrder', 'additional_batch_order', 'additional_data_iteration', 'amp']:
        if key_item in checkpoint:
            del checkpoint[key_item]

    print("after deleting, the keys are:", checkpoint.keys())
    save_checkpoint = checkpoint["model"]

    print("save the extracted model to:", output)
    torch.save(save_checkpoint, output)

if __name__ == "__main__":
    main()
