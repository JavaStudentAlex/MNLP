from argparse import ArgumentParser
import lightning.pytorch as pl
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# Lightning: https://lightning.ai/docs
# Huggingface: https://huggingface.co/
# WandB: https://docs.wandb.ai/
# Additional: https://github.com/ashleve/lightning-hydra-template

# Define the our Model


class MLP(torch.nn.Module):
    def __init__(self, embedding_dim=2, block_size=3, hidden_dim=100, vocab_size=27, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.cat_dim = embedding_dim * block_size
        self.C = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dense = torch.nn.Linear(embedding_dim*block_size, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.C(x)
        h = self.dense(emb.view(-1, self.cat_dim))
        h = F.tanh(h)
        logits = self.out(h)
        return logits

# Define the Lightning Module (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule)


class MLPModule(pl.LightningModule):
    def __init__(self, embedding_dim=2, block_size=3, hidden_dim=100, vocab_size=27):
        # Save HP to checkpoints
        self.save_hyperparameters()
        super().__init__()
        # Init model
        self.model = MLP(embedding_dim, block_size, hidden_dim, vocab_size)

    # def setup(self, stage):
        # Change your model dynamically

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log_dict({"val_loss": loss}, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log_dict({"test_loss": loss}, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True)

    # def predict_step(self, batch, batch_idx):

    # Returns an optimizer or a dictionary with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.1)


class NameDataset(torch.utils.data.Dataset):
    def __init__(self, words, stoi, block_size) -> None:
        super().__init__()
        self.X, self.Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                self.X.append(context)
                self.Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] if self.Y is not None else self.X[idx]

# Lightning Datamodule (https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule)


class NamesDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "names.txt", block_size=3):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size

    # def prepare_data(self):
        # Download data

    def setup(self, stage):
        # Load data
        words = open('names.txt', 'r').read().splitlines()
        chars = sorted(list(set(''.join(words))))
        stoi = {s: i+1 for i, s in enumerate(chars)}
        stoi['.'] = 0

        # Split
        random.seed(42)
        random.shuffle(words)
        n1 = int(0.8*len(words))
        n2 = int(0.9*len(words))

        # Create datasets
        self.train_dataset = NameDataset(words[:n1], stoi, self.block_size)
        self.validation_dataset = NameDataset(
            words[n1:n2], stoi, self.block_size)
        self.test_dataset = NameDataset(words[n2:], stoi, self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=32, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False)
    # HINT: Evaluating on multiple dataloaders (https://lightning.ai/docs/pytorch/LTS/guides/data.html)

# Training and evaluation script


def main(hparams):
    # Init logging
    wandb_logger = WandbLogger(
        log_model=False, project="MNLP", name=f"SimpleMLP")

    # Init the building blocks
    module = MLPModule()
    datamodule = NamesDataModule()
    # Trainer (https://lightning.ai/docs/pytorch/stable/common/trainer.html)
    trainer = pl.Trainer(accelerator=hparams.accelerator,
                         devices=hparams.devices,
                         max_epochs=2,
                         logger=wandb_logger)

    # Fit the model (and evaluate on validation data as defined)
    trainer.fit(module, datamodule=datamodule)

    # Test model
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
