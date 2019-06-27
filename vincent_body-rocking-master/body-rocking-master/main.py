# File: main.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:24 PM
import os
import pickle

import torch
from torch import nn

from config import RANDOM_SEED, TRAIN_EPOCHS, MODEL_DIR, MODEL_PREFIX
from dataset_utils import load_datasets
from rocking_dataset import RockingDataset
from training_utils import get_model, fit

torch.manual_seed(RANDOM_SEED)


def save_loss_acc(train_loss, train_acc, val_loss, val_acc):
    pickle.dump(train_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_train_loss.pkl"), "wb"))
    pickle.dump(train_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_train_acc.pkl"), "wb"))
    pickle.dump(val_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_val_loss.pkl"), "wb"))
    pickle.dump(val_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_val_acc.pkl"), "wb"))


dataset_loaders, dataset_sizes = load_datasets(RockingDataset)
train_dataloader = dataset_loaders['train']
val_dataloader = dataset_loaders['val']

model, opt = get_model()
criterion = nn.CrossEntropyLoss()

# from LRFinder import LRFinder
# lr_finder = LRFinder(model, opt, criterion, device="cpu")
# lr_finder.range_test(train_dataloader, val_dataloader, end_lr=0.1, num_iter=100)
# lr_finder.plot()

train_loss, train_acc, val_loss, val_acc = fit(TRAIN_EPOCHS, model, criterion, opt, train_dataloader, val_dataloader)

save_loss_acc(train_loss, train_acc, val_loss, val_acc)