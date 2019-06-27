# File: predict.py
# Author: Ronil Pancholia
# Date: 3/25/19
# Time: 10:34 PM

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from config import RANDOM_SEED, device, MODEL_DIR
from dataset_utils import load_testset
from rocking_dataset import RockingDataset
from training_utils import get_model

torch.manual_seed(RANDOM_SEED)


def test_model(model):
    test_dataloader, test_datasize = load_testset(RockingDataset)
    model.eval()
    results = []
    for image, label in tqdm(test_dataloader['test'], file=sys.stdout):
        image = image.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    return results


model, opt = get_model()
model_name = sys.argv[1]
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_name)))

results = test_model(model)

with open("prediction.txt", "w") as f:
    for r in results:
        f.write(str(r) + "\n")
