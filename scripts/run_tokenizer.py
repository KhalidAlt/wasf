import argparse, logging
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer,trainers,pre_tokenizers
from tokenizers.processors import BertProcessing

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
        type=str,
        default=50_000,
        help ='vocabulary size of the tokenizer'
    )
    Parser.add_argument(
        '--min_freq',
        type=str,
        default=2,
        help ='minimum frequence to a word to be saved in the vocab'
    )
    Parser.add_argument(
        "--batch_size",
        type=str,
        default=1000,
        help='batch size'
    )
    args = Parser.parse_args()
    return args


def main(argv):

    ds = load_dataset(args.dataset_name,args.subset, cache_dir=args.cache_dir)
    tokenizer = ByteLevelBPETokenizer(lowercase=True)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(ds), batch_size):
            yield ds['train'][i : i + batch_size]["text"]


    trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size, min_frequency=args.min_freq,
            special_tokens=["<s>", "<pad>", "</s>",
                            "<unk>", "<mask>"])
    print("Start Training ...")
    tokenizer.train_from_iterator(batch_iterator(), length=len(ds))

    print("Done training")


#if __name__ == '__main__':
args = get_args()

main(args)
