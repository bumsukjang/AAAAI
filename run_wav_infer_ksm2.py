import time
import argparse
import os    
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from fastprogress.fastprogress import progress_bar
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from tqdm import tqdm
from Google_STT_multiCore import GoogleSTT
from transformers import ElectraTokenizer, ElectraForSequenceClassification
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#%%
def load_model(device):
    checkpoint = os.path.dirname(os.path.realpath(__file__))+'/ckpt/koelectra-base-v3-ckpt1/checkpoint-1300'
    model = ElectraForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    
    print('model loaded')
    return model
#%%    
def evaluate(args, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    preds = None
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = { "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3] }
            inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)

    return preds
#%%
def main(args):

    GoogleSTT(args.in_wav_folder)

    model=load_model(device)
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", do_lower_case=False)

    if os.path.exists(args.out_infer_file):
        os.remove(args.out_infer_file)

    out_infer_folder = os.path.dirname(args.out_infer_file)
    if not os.path.exists(out_infer_folder): 
        os.makedirs(out_infer_folder)
        
    wf =  open(args.out_infer_file, "a")

    files = os.listdir('tmp_ASR')   
    for filename in tqdm(files):
        if os.path.splitext(filename)[1].lower() == '.txt':
            fullPath = "{}\{}".format('tmp_ASR', filename).replace('\\', '/')

            args.infer_file = fullPath
            test_dataset  = load_and_cache_examples(args, tokenizer, mode="infer") 
            preds = evaluate(args, model, test_dataset)

            filename=filename.split('.txt')[0]
            text = filename + ':' + str(preds[0])
            print(text, file=wf)
            
def pred_sm(args):
    model=load_model(device)
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", do_lower_case=False)

    if os.path.exists(args.out_infer_file):
        os.remove(args.out_infer_file)

    out_infer_folder = os.path.dirname(args.out_infer_file)
    if not os.path.exists(out_infer_folder): 
        os.makedirs(out_infer_folder)
        
    wf =  open(args.out_infer_file, "a")
    resultData = []
    files = os.listdir(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR')   
    for filename in tqdm(files):
        if os.path.splitext(filename)[1].lower() == '.txt':
            #fullPath = "{}\{}".format('tmp_ASR', filename).replace('\\', '/')
            fullPath = filename
            args.infer_file = fullPath
            test_dataset  = load_and_cache_examples(args, tokenizer, mode="infer") 
            preds = evaluate(args, model, test_dataset)

            filename=filename.split('.txt')[0]
            text = filename + ':' + str(preds[0])
            print(text, file=wf)
            resultData.append({
                filename: str(preds[0])
            })
    return resultData
#%%    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--out_infer_file", type=str, default='./infer_result/infer_results.txt', dest='out_infer_file')      
    parser.add_argument("--in_wav_folder",  type=str, default='./Test_DB_wav',                    dest='in_wav_folder')      
    
    cli_args = parser.parse_args()
    
    start_time = time.time()
    main(cli_args)
    print('elapsed time:', time.time() - start_time)