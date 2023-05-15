import re
import random
import numpy as np
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from dl.bengio2003.litmodule import NeuralProbabilisticLanguageModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


BATCH_SIZE = 128
CONTEXT = 8
HIDDEN_UNITS = 100
FEATURES = 2


class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataset(words: list[str], wtoi: dict):
    X, Y = [], []
    context = [0] * CONTEXT
    for w in words:
        ix = wtoi[w]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X[3:], Y[3:]


def plot_2d_embeddings(embeddings, itow):
    words = [itow[i] for i in range(100)]

    plt.figure(figsize=(15, 15))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(embeddings[i, 0], embeddings[i, 1]))

    plt.show()


def main():
    data = open("data/shakespeare.txt", "r").read().splitlines()
    data = [d for d in data if d != ""]
    data = [re.sub(r"[^a-zA-Z0-9 ]", "", d) for d in data]
    data = [d.split(" ") for d in data]
    data = [d for d in data if len(d) > 1]
    data = [item for sublist in data for item in sublist]
    data = [d.split(".") for d in data]
    data = [item for sublist in data for item in sublist]
    data = [d if d != "" else "." for d in data]
    data = [d.lower() for d in data]
    data = np.array(data)

    data = data[:10_000]

    # vocabulary V
    vocabulary = {}

    for w in data:
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1

    wtoi = {w: i for (i, w) in enumerate((list(vocabulary.keys())))}
    itow = {value: key for (key, value) in wtoi.items()}

    random.seed(42)
    random.shuffle(data)
    n1 = int(0.8 * len(data))
    n2 = int(0.9 * len(data))

    Xtr, Ytr = build_dataset(data[:n1], wtoi)
    Xdev, Ydev = build_dataset(data[n1:n2], wtoi)
    Xte, Yte = build_dataset(data[n2:], wtoi)

    model = NeuralProbabilisticLanguageModel(
        batch_size=BATCH_SIZE,
        hidden_units=HIDDEN_UNITS,
        context=CONTEXT,
        features=FEATURES,
        vocab_size=len(vocabulary),
        seed=234245,
    )

    shakespeare_tr = ShakespeareDataset(X=Xtr, y=Ytr)
    shakespeare_val = ShakespeareDataset(X=Xdev, y=Ydev)

    loader_tr = DataLoader(
        shakespeare_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    loader_val = DataLoader(shakespeare_val, batch_size=BATCH_SIZE, num_workers=8)

    callbacks = [
        TQDMProgressBar(refresh_rate=1_000),
    ]

    trainer = Trainer(max_epochs=30, callbacks=callbacks)

    trainer.fit(model, train_dataloaders=loader_tr, val_dataloaders=loader_val)

    for _ in range(10):
        out = model.predict()
        txt = [itow[ix] for ix in out]
        print(txt)

    C = model.model.C.detach().numpy()

    embeddings = C[:100]

    if FEATURES == 2:
        plot_2d_embeddings(embeddings, itow)


if __name__ == "__main__":
    main()
