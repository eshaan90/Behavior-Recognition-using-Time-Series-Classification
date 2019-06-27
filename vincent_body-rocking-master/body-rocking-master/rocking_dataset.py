# File: rocking_dataset.py
# Author: Ronil Pancholia
# Date: 3/7/19
# Time: 2:50 AM
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from read_data import read_session_data


class RockingDataset(Dataset):
    def __init__(self, dir_path, mode, transforms=None):
        # Set transforms
        self.transforms = transforms

        arm_files = [os.path.join(os.path.join(dir_path, session), "armIMU.txt") for session in os.listdir(dir_path)]
        wrist_files = [os.path.join(os.path.join(dir_path, session), "wristIMU.txt") for session in os.listdir(dir_path)]
        detection_files = [os.path.join(os.path.join(dir_path, session), "detection.txt") for session in os.listdir(dir_path)]

        print("Loading dataset from ", dir_path)
        self.arm_data = read_session_data(arm_files, multi_value=True, is_label=False, mode=mode)
        print("Loaded arm data :", self.arm_data.shape)
        self.wrist_data = read_session_data(wrist_files, multi_value=True, is_label=False, mode=mode)
        print("Loaded wrist data :", self.wrist_data.shape)

        self.mode = mode
        if mode != "test":
            self.label_arr = read_session_data(detection_files, multi_value=False, is_label=True, mode=mode)
            print("Loaded label data :", self.label_arr.shape)

        # Calculate len
        self.data_len = len(self.arm_data)

    def __getitem__(self, index):
        arm = self.arm_data[index]
        wrist = self.wrist_data[index]
        input = np.concatenate([arm, wrist], axis=1)

        if self.transforms is not None:
            input = self.transforms(input)

        # Transform image to tensor
        input_tensor = torch.Tensor(input)

        # Get label of the image
        if self.mode == "test":
            label_tensor = torch.tensor(0)
        else:
            label_tensor = torch.tensor(self.label_arr[index])

        return (input_tensor, label_tensor)

    def __len__(self):
        return self.data_len
