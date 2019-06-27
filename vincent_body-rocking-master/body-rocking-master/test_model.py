# File: test_model.py
# Author: Ronil Pancholia
# Date: 3/24/19
# Time: 12:50 AM

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from config import RANDOM_SEED, device, MODEL_DIR, FREQUENCY, TIME_WINDOW
from dataset_utils import load_testset
from rocking_dataset_eval import RockingDatasetEval
from training_utils import get_model

torch.manual_seed(RANDOM_SEED)

def test_model(model):
    test_dataloader, test_datasize = load_testset(RockingDatasetEval)
    model.eval()
    results = []
    corrects = 0
    for image, label in tqdm(test_dataloader['test'], file=sys.stdout):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        raw_preds = torch.argmax(outputs.data, 1)
        preds = torch.Tensor(np.asarray(raw_preds).astype(np.long))
        labels = []
        for l in label:
            labels.extend(l)
        labels = torch.Tensor(np.asarray(labels).astype(np.long))
        corrects += torch.sum(preds == labels).item()
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    print("Testing accuracy :", (100.0 * corrects) / (len(test_dataloader['test'].dataset) * FREQUENCY * TIME_WINDOW))

model, opt = get_model()
model_name = sys.argv[1]
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_name)))

test_model(model)