# File: plot_curves.py
# Author: Ronil Pancholia
# Date: 3/24/19
# Time: 1:09 AM

import matplotlib.pyplot as plt
import numpy as np
import config
import pickle


def save_plots(train_loss, train_acc, val_loss, val_acc):
    lst_iter = np.arange(1, len(train_loss) + 1)
    plt.plot(lst_iter, train_loss, '-b', label='training loss')
    plt.plot(lst_iter, val_loss, '-r', label='validation loss')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(config.MODEL_PREFIX + "_loss.png")
    plt.clf()

    plt.plot(lst_iter, train_acc, '-b', label='training accuracy')
    plt.plot(lst_iter, val_acc, '-r', label='validation accuracy')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='bottom right')
    # plt.show()
    plt.savefig(config.MODEL_PREFIX + "_accuracy.png")

train_acc = pickle.load(open("cnn_v3_train_acc.pkl", "rb"))
val_acc = pickle.load(open("cnn_v3_val_acc.pkl", "rb"))
train_loss = pickle.load(open("cnn_v3_train_loss.pkl", "rb"))
val_loss = pickle.load(open("cnn_v3_val_loss.pkl", "rb"))

save_plots(train_loss, train_acc, val_loss, val_acc)