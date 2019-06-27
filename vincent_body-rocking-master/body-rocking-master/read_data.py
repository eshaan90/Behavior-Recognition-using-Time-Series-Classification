# File: read_data.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 4:43 PM
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import TIME_WINDOW, FREQUENCY, TRAIN_OVERLAP, TEST_OVERLAP

val_bucket_length = 0
train_bucket_length = 0
def create_buckets(data, mode, is_label):
    global val_bucket_length, train_bucket_length
    overlap = TEST_OVERLAP if mode == "test" else TRAIN_OVERLAP

    step = int((FREQUENCY * TIME_WINDOW) * overlap)
    size = FREQUENCY * TIME_WINDOW
    if mode in ["test", "val"]:
        step = 1
        if not is_label:
            zero_vector = [0, 0, 0, 0, 0, 0]
            data = [zero_vector]* (size - 1) + data
        else:
            zero_vector = [0]
            data = [zero_vector] * (size - 1) + data
            buckets = [data[i] for i in range(size, len(data), step)]
            if mode == "val":
                for i in range(val_bucket_length - len(buckets)):
                    buckets.append(0)

            return buckets

    buckets = [data[i: i + size] for i in range(0, len(data), step)]
    count = 0
    for i, b in enumerate(buckets):
        if (len(b) != size):
            count += 1

    buckets = buckets[:-count]
    if mode == "val":
        val_bucket_length = len(buckets)
    else:
        train_bucket_length = len(buckets)

    return buckets


def compute_test_labels(buckets):
    labels = []
    for bucket in buckets:
        temp = []
        for label in bucket:
            temp.append(label)
        labels.append(temp)

    labels = [np.asarray(x) for x in labels]

    return np.asarray(labels).astype(np.long)


def compute_train_labels(buckets):
    labels = []
    for bucket in buckets:
        zeros = 0
        for label in bucket:
            if label == 0:
                zeros += 1

        if zeros > 0.5 * len(bucket):
            labels.append(0)
        else:
            labels.append(1)

    return np.asarray(labels).astype(np.long)


def read_session_data(files, multi_value=False, is_label=False, mode="train"):
    session_data = []
    for file_name in files:
        with open(file_name, "r") as f:
            for line in f:
                if multi_value:
                    values = [float(x) for x in line.strip().rsplit("  ")]
                else:
                    values = float(line.strip())

                session_data.append(values)

    # pickle_file = file_name.split("/")[-1][:-4] + "_scalar.pkl"
    # if mode in ["train"] and not is_label:
    #     scalar = StandardScaler()
    #     scalar.fit(session_data)
    #     pickle.dump(scalar, open(pickle_file, "wb"))
    # elif not is_label:
    #     scalar = pickle.load(open(pickle_file, "rb"))
    #
    # if not is_label:
    #     session_data = scalar.transform(session_data)

    buckets = create_buckets(session_data, mode, is_label)

    buckets = [np.asarray(x) for x in buckets]

    if is_label:
        if mode == "test":
            buckets = compute_test_labels(buckets)
        elif mode == "val":
            buckets = np.asarray(buckets).astype(np.long)
        else:
            buckets = compute_train_labels(buckets)

    return np.asarray(buckets)
