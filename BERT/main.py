import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, BertTokenizerFast, AutoModel, AutoTokenizer

from model_loader import BSENTClassificationH
import dataset
import train


def define_argparser():
    now = datetime.now()
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default=f"{now.strftime('%Y-%m-%d %H:%M:%S')}")

    p.add_argument('--data', default="../data/")  # dataset directory
    p.add_argument('--model_fn', default=f"./save_model/model.pt")  # model save path
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--p', type=float, default=0.1, help="Dropout rate")
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=5)

    config = p.parse_args()
    return config


def main(config):

    # device
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    print(f"Working: {device}")

    # load tokenizer, bert, model
    # bert_model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    # tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)
    bert_model = AutoModel.from_pretrained("snunlp/KR-Medium")
    model = BSENTClassificationH(device=device, model=bert_model).to(device)

    # optimizer, scheduler, loss function
    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95**x)
    loss_fn = nn.CrossEntropyLoss()

    # load dataset
    data_train = dataset.CustomDataset(config.data+"test.pkl", tokenizer=tokenizer)
    data_test = dataset.CustomDataset(config.data+"data30000.pkl", tokenizer=tokenizer)
    train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=1)
    test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=1)

    # load trainer, train start
    trainer = train.Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler)
    trainer.train(train_data=train_dataloader, test_data=test_dataloader, device=device, config=config)

    # if directory is not exist, make save model directory
    if not os.path.exists("./save_model"):
        os.makedirs("./save_model")

    # save model
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config,
    }, config.model_fn)
    # torch.save(trainer.best_model, config.model_fn)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
