import torch


def calc_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.detach().cpu().numpy()/max_indices.size()[0]
    return train_acc


def f1_score(x, y):
    pass

