import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, transform=True, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len # max len - special token(cls, sep)
        self.transform = transform

        sentiment_cls = {
            "감정없음": 0,
            "놀람": 1,
            "평온함(신뢰)": 2,
            "기쁨": 3,
            "슬픔": 4,
            "불안": 5,
            "분노": 6
        }

        # self.text_emb = pd.read_csv(path)["text"].apply(lambda x: tokenizer.encode(x))
        self.text_emb = pd.read_pickle(path)["text_script"]
        self.label = pd.read_pickle(path)["CML 감정"].apply(lambda x: sentiment_cls[x])
        self.mfcc = pd.read_pickle(path)["audio"]

    def __len__(self):
        return len(self.text_emb)

    def __getitem__(self, idx):
        x = self.text_emb.iloc[idx]
        x = self.tokenizer.encode(x)
        label = self.label.iloc[idx]
        mfcc = self.mfcc.iloc[idx]

        if len(x) < self.max_len:
            x = x + [0]*(self.max_len-len(x))
        else:
            x = x[:self.max_len]

        if len(mfcc) < 6000:
            # mfcc = mfcc + [0]*(6000-len(x))
            mfcc = np.append(mfcc, np.array([-100]*(6000-len(mfcc))))
        else:
            mfcc = mfcc[:6000]

        pad_x = [1 if i >= 1 else 0 for i in x]
        if self.transform:
            x = torch.tensor(x)
            pad_x = torch.tensor(pad_x)
            label = torch.tensor(label)
            mfcc = torch.tensor(mfcc)
        return x, pad_x, label, mfcc
