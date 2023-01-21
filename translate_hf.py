import argparse, logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,M2M100Config,M2M100Tokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm 
import torch

def get_args():
    Parser = argparse.ArgumentParser(description="Machine Translation Evalution")
    Parser.add_argument(
        '--model_name_or_path',
        type=str,
        help = 'the name or path of the model to use in the test.',
        required=True
    )
    Parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        help = 'the name or path of the tokenizer to use in the test',
        required=True
    )
    Parser.add_argument(
        '--dataset',
         type=str,
          help = 'The name of the dataset to use for test.',
           required=True
       )
    Parser.add_argument(
        '--split',
        type=str,
        help ='dataset split to be use in the inference'
    )
    Parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help ='dataset subset to be use'
    )
    Parser.add_argument(
        "--search_method",
        type=str,
        default='beam',
        help='Choose decoding method'
    )
    Parser.add_argument(
        "--cache_dir",
        type=str,
        default = '/media/khalid/data_disk/cache/',
        help = ''
    )
    Parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help='the device in which the test will run on.'
    )
    Parser.add_argument(
        '--column_name',
        type=str,
        help = 'target language code.'
    )
    Parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help = 'Use this argument if you want to rename the columns of the dataset.'
    )
    Parser.add_argument(
        '--rename',
        type=str,
        default=True,
        help = 'Use this argument if you want to rename the columns of the dataset.'
    )

    Parser.add_argument(
        '--save_translation',
        type=str,
        default= True,
        help = 'Use this argument if you want to save the translation.'
    )
    Parser.add_argument(
        '--evaluate',
        type=str,
        default= True,
        help = 'Use this argument if you want to evaluate the generarted translation.'
    )

    Parser.add_argument(
        '--save_path',
        type=str,
        help = 'Use this argument if you want for the path where you want to save the result.'
    )
    Parser.add_argument(
        '--toy_example',
        type=bool,
        default=False,
        help = 'Use this arguments if you want to use only 10 examples.'
    )
    Parser.add_argument(
        '--push_to_hub',
        type=str,
        default=True,
        help = 'Use this arguments if you want to push your dataset to HuggingFace Hub.'
    )
    args = Parser.parse_args()
    return args



def main(argv):




    if '.csv' in args.dataset or '.tsv' in args.dataset:
        ds = load_dataset('csv',
                  delimiter='\t',
                  header=None,
                  split=args.split,
                  data_files=[args.dataset],
                 cache_dir=args.cache_dir)
        
        columns_name = ds.column_names
        if args.rename == True:

            assert len(columns_name)==2, 'The number of columns must be two'
            ds = ds.rename_columns({columns_name[0]:'image_alt', columns_name[1]:'url'})

    else:
        
        ds = load_dataset(args.dataset,args.subset,split=args.split,cache_dir=args.cache_dir)

    if args.toy_example:

        ## This line to be removed
        ds = ds.select(range(10))
    
    train_dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)




    translated = []
    for i in tqdm(train_dataloader):
        inputs=tokenizer(i[args.column_name],return_tensors='pt',truncation=True,padding='longest')
        inputs['input_ids'] = inputs['input_ids'].to(args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(args.device)

        inputs['input_ids'] = inputs['input_ids'].to(args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(args.device)

        translated_tokens = model.generate(
        **inputs.to(args.device), forced_bos_token_id=tokenizer.lang_code_to_id['ar'], max_length=30,num_beams=3,
        )
        
        outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


        [translated.append([i1, i2,i3]) for i1,i2,i3 in zip(i[args.column_name],outputs,i['image_url'])] 


        if len(translated) % 100_000 == 0:
                    
            df = pd.DataFrame(translated, columns=['caption','ar','url'])
            df.to_json(f'{args.save_path}/ccg.json',
                        force_ascii=False,
                        orient='records',
                        lines=True)
        
            if args.push_to_hub == True:
                ds = Dataset.from_pandas(df)
                ds.push_to_hub('khalidalt/ccg_arabic', private=True)
                translated = []

                    


#if __name__ == '__main__':
args = get_args()
print("Here")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,cache_dir=args.cache_dir)
print("Tokenizer Build Successfully")
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,cache_dir=args.cache_dir)
print("Model Build Successfully")
print(model)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.device == 'cuda' and device =='cuda':
    
    model.to(device)

elif args.device == 'cuda':
    print("Your Device does not show any cuda devices!, CPU will be use instead")

main(args)
