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

    ds = load_dataset(args.dataset_name,args.subset,split='train', cache_dir=args.cache_dir)
    tokenizer = ByteLevelBPETokenizer(lowercase=True)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(ds), batch_size):
            yield ds[i : i + batch_size]["text"]


    print("Start Training ...")
    tokenizer.train_from_iterator(batch_iterator(),
                        vocab_size=args.vocab_size,
                        min_frequency=args.min_freq,
                        length=len(ds),)

    print("Done training")
    tokenizer.save_model(args.output_path)

    # Save the configuration of the model (config.json)
    config_file = AutoConfig.from_pretrained(args.config_name)
    config_file.save_pretrained(args.output_path)

    # save the tokenizer files
    print(f"The tokenizer saved in {args.output_path}")


#if __name__ == '__main__':
args = get_args()

main(args)
