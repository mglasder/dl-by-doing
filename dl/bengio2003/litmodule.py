import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from dl.bengio2003.modules import NPLM


class NeuralProbabilisticLanguageModel(pl.LightningModule):
    def __init__(
        self, batch_size, hidden_units, context, features, vocab_size, seed=42
    ):
        super().__init__()

        self.n = context
        self.g = torch.Generator().manual_seed(seed)

        self.model = NPLM(
            batch_size, hidden_units, context, features, vocab_size, self.g
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def predict(self):
        out = []
        context = [0] * self.n
        while True:
            logits = self.model(torch.tensor(context))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=self.g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 10:
                break

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2,
            factor=0.1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
