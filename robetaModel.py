import os
import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.fcn import FCN
from data_loader import RoBertaDataset
#from utils import monte_carlo_dropout, compute_confidnce_interval

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

FCN_MODEL_PATH = os.path.dirname(os.path.realpath(__file__))+'/models/roberta-base-classi-model_20201118_6-class.pt'



def load_model( modelpath, model):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state)

    print('model loaded')


def main(args):
    roberta_dataset = RoBertaDataset()

    fcn_model = FCN(768).to(DEVICE)
    load_model(FCN_MODEL_PATH, fcn_model)
    fcn_model.eval()
    fcn_model = monte_carlo_dropout(fcn_model)

    pred_dl = DataLoader(roberta_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(pred_dl):
        total_output = []
        for i in range(512):
            output = fcn_model(samples)
            total_output.append(output.detach().cpu().numpy())
        total_output = np.array(total_output)
        output_mean = np.mean(total_output, axis=0)
        pred = np.argmax(output_mean, axis=1)
        pred = compute_confidnce_interval(pred, output_mean)

        print(output_mean.shape, pred.shape)
        print(pred)

def pred_cj(args):
    listFile=os.path.dirname(os.path.realpath(__file__))+'/text/asr_result_list.txt'
    with open(listFile, 'rt', -1, 'utf-8') as rf:
        fileList = list(map(lambda line : line.strip(), rf.readlines()))
    
    roberta_dataset = RoBertaDataset()

    fcn_model = FCN(768).to(DEVICE)
    load_model(FCN_MODEL_PATH, fcn_model)
    fcn_model.eval()
    #fcn_model = monte_carlo_dropout(fcn_model)

    pred_dl = DataLoader(roberta_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred_result = None
    for batch_idx, samples in enumerate(pred_dl):

        output = fcn_model(samples)
        pred = output.max(1, keepdim=True)[1]
        pred = pred.detach().cpu().numpy()
        pred[pred == 5] = 0
        if pred_result is None:
            pred_result = pred
        else:
            pred_result = np.concatenate((pred_result, pred), axis=0)

    return fileList, pred_result
        
        #total_output = []
        #for i in range(512):
        #    output = fcn_model(samples)
        #    total_output.append(output.detach().cpu().numpy())
        #total_output = np.array(total_output)
        #output_mean = np.mean(total_output, axis=0)
        #pred = np.argmax(output_mean, axis=1)
        #pred = compute_confidnce_interval(pred, output_mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of mini batch size',
        default=64,
        type=int)

    args = parser.parse_args()
    print(args)

    start_time = time.time()
    result_cj = pred_cj(args)
    print("result(cj) : ", result_cj)
    print('elapsed time:', time.time() - start_time)