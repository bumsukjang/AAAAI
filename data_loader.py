import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from fairseq.models.roberta import XLMRModel

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

class RoBertaDataset(Dataset):
    def __init__(self, textpath='/text/asr_result.txt'):
        self.textpath = os.path.dirname(os.path.realpath(__file__))+textpath

        self.xlmr = XLMRModel.from_pretrained('xlmr.base', checkpoint_file='model.pt')
        self.xlmr = self.xlmr.to(DEVICE)
        self.xlmr.eval()
        print('xlmr base model loaded.')

        with open(self.textpath, 'rt', -1, 'utf-8') as rf:
            self.data = rf.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.xlmr.encode(self.data[idx])
        last_layer_features = self.xlmr.extract_features(tokens)
        last_layer_features = torch.squeeze(last_layer_features)

        return last_layer_features[0]


if __name__ == "__main__":
    roberta_dataset = RoBertaDataset()
    print(roberta_dataset)

    data_loader = DataLoader(roberta_dataset, batch_size=4, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(data_loader):
        data = samples

        print(data[0])
        print(data[1])
        print(data[2])
        print(data[3])

        break