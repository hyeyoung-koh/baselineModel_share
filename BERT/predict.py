import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import model_loader
import numpy as np
import pandas as pd
import dataset
from tqdm import tqdm


def predict(path, bs, tokenizer_, data_path, device=torch.device("cuda")):
    result = []
    model = torch.load(path).to(torch.device(device))
    data_predict = dataset.CustomDataset(data_path, tokenizer=tokenizer_)
    predict_data = DataLoader(data_predict, batch_size=bs, num_workers=5)
    with tqdm(total=len(predict_data)) as pbar:
        for step, (x, pad_x, _) in enumerate(predict_data):
            x = x.cuda()
            res = model(x, pad_x).to(device)
            max_vals, max_indices = torch.max(res, 1)
            result.append(list(max_indices.detach().cpu()))
            pbar.update(1)
    return torch.flatten(torch.tensor(result))


if __name__ == "__main__":
    PATH = "save_model/model0427.pt"
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium")
    res = predict(PATH, 16, tokenizer, "../data/test.csv")
    d = pd.read_csv("../data/test.csv")
    d["predict"] = res
    d.to_csv("./pred/result.csv", encoding="utf-8-sig")

