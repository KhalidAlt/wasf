


import argparse, logging
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer,trainers,pre_tokenizers
from tokenizers.processors import BertProcessing
from transformers import AutoConfig

def get_args():
    Parser = argparse.ArgumentParser(description="Machine Translation Evalution")
    Parser.add_argument(
        '--dataset_name',
        type=str,
        help = 'the name or path of the model to use in the test.',
        required=True
    )
    Parser.add_argument(
        '--subset',
        type=str,
        help = 'the name or path of the subset of the dataset to use in the test',
        required=True
    )
    Parser.add_argument(
        '--cache_dir',
         type=str,
          help = 'The directory of the cache where the dataset is saved.',
           required=True
       )
    Parser.add_argument(
        '--vocab_size',
        type=int,
        default=50265,
        help ='vocabulary size of the tokenizer'
    )
    Parser.add_argument(
        '--min_freq',
        type=int,
        default=2,
        help ='minimum frequence to a word to be saved in the vocab'
    )
    Parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help='batch size'
    )
    Parser.add_argument(
        "--output_path",
        type=str,
        help='output path where the tokenizer will be saved'
    )
    Parser.add_argument(
        "--config_name",
        type=str,
        help='the name of the model configuration'
    )
    args = Parser.parse_args()
    return args


def main(argv):

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              use_fast=True,
                                              model_max_length=512)  # model_max_length=512

    # Load the dataset
    ds = load_dataset(args.dataset_name,args.subset,split='train', cache_dir=args.cache_dir)

    # Define a function to tokenize an example and return the number of tokens
    num_tokens = []
    # Define a function to tokenize an example and return the number of tokens
    def count_tokens(example):
        tokens = tokenizer.encode_batch(example['text'])
        num_tokens.append(sum([len(i.tokens) for i in tokens]))
        return example


    # Print the result
    print("The dataset contains", sum(num_tokens), "tokens.")



#if __name__ == '__main__':
args = get_args()

main(args)
