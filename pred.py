import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.fcn import FCN
from data_loader import RoBertaDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

FCN_MODEL_PATH = './models/roberta-base-classi-model_20201118_6-class.pt'



def load_model( modelpath, model):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state)

    print('model loaded')


def main(args):
    roberta_dataset = RoBertaDataset()

    fcn_model = FCN(768).to(DEVICE)
    load_model(FCN_MODEL_PATH, fcn_model)
    fcn_model.eval()
    #fcn_model = monte_carlo_dropout(fcn_model)

    pred_dl = DataLoader(roberta_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(pred_dl):
        output = fcn_model(samples)
        pred = output.max(1, keepdim=True)[1]
        pred = pred.detach().cpu().numpy()
        pred[pred == 5] = 0

        print(pred.shape, np.unique(pred))





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
    main(args)
    print('elapsed time:', time.time() - start_time)