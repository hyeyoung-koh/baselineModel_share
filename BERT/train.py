from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
import metrics


class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.best_model = None
        super().__init__()

    def _train(self, train_data, device, epoch):
        self.model.train()
        train_acc = 0.0
        train_loss = 0.0
        count = 0
        with tqdm(total=len(train_data), desc=f"Train-{epoch}") as pbar:
            for step, (x, pad_x, label, mfcc) in enumerate(train_data):
                self.model.zero_grad()

                x = x.long().to(device)
                label = label.long().to(device)

                y_hat = self.model(x, pad_x, mfcc).to(device)
                loss = self.loss_fn(y_hat, label)

                train_acc += metrics.calc_accuracy(y_hat, label)
                train_loss += loss.detach().cpu()

                loss.backward()
                self.optimizer.step()
                pbar.update(1)
                count += 1
                pbar.set_postfix_str(
                    f"Acc: {train_acc / count: .3f} Loss: {loss.item():.3f} ({train_loss / count:.3f})"
                )
            if self.scheduler:
                self.scheduler.step()
            learning_rate = self.scheduler.get_last_lr()[0]

            return train_loss / count, 100 * (train_acc / count), learning_rate

    def _validate(self, test_data, device, epoch):
        self.model.eval()

        test_acc = 0.0
        test_loss = 0.0
        test_count = 0
        with torch.no_grad():
            with tqdm(total=len(test_data), desc=f"Validate-{epoch}") as pbar:
                for step, (x, pad_x, label, mfcc) in enumerate(test_data):
                    x = x.long().to(device)
                    mfcc = mfcc.long().to(device)
                    label = label.long().to(device)

                    y_hat = self.model(x, pad_x, mfcc).to(device)
                    loss = self.loss_fn(y_hat, label)
                    test_loss += loss.detach().cpu()
                    test_acc += metrics.calc_accuracy(y_hat, label)
                    test_count += 1
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"Acc: {test_acc / test_count: .3f} Loss: {loss.item():.3f} ({test_loss / test_count:.3f})"
                    )
        return test_loss/test_count, 100 * (test_acc / test_count)

    def train(self, train_data, test_data, device, config=None):
        lowest_loss = np.inf
        highest_acc = 0.0
        best_model = None

        for epoch in range(config.n_epochs):
            train_loss, train_acc, learning_rate = self._train(train_data, device, epoch)
            valid_loss, valid_acc = self._validate(test_data, device, epoch)

            if valid_loss <= lowest_loss:
                # best_model = deepcopy(self.model.state_dict())
                best_model = deepcopy(self.model)
                print("Epoch(%d/%d): train_loss=%.4f valid_loss=%.4f before_lowest_loss=%.4f" % (
                    epoch + 1, config.n_epochs, train_loss, valid_loss, lowest_loss
                ))
                lowest_loss = valid_loss

        self.best_model = best_model
