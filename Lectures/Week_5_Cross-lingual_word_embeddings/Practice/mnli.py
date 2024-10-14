# typical PyTorch imports
from typing import Any, cast
from lightning.pytorch.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

# there's a plethora of AutoModel{ForTypeOfTask}
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import partial

from lightning.pytorch import Trainer, LightningModule
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Useful links:
# Huggingface: https://huggingface.co/
# - Models, Datasets
# Huggingface examples: https://github.com/huggingface/transformers/tree/main/examples/pytorch
# - Data (preprocessing, loading, evaluation) and modelling for all sorts of tasks
# Lightning: https://lightning.ai/docs
# - Clean training and evaluation loop with easy customizability (Huggingface transformers also has a similar `Trainer` class)
# Evaluation metrics
# - torchmetrics: https://lightning.ai/docs/torchmetrics/stable/ - clean functional API for simple metrics
# - evaluate:  https://github.com/huggingface/evaluate/ - supports all sorts of involved metrics for generation or token-level tasks
# WandB: https://docs.wandb.ai/
# - Clean logging to an online dashboard
# Additional: https://github.com/ashleve/lightning-hydra-template

# CONSTANTS
# pretrained_model_name_or_path is a common variable name in Hugging Face to point to the model in the cloud (model_name) 'or' locally (path)
pretrained_model_name_or_path = (
    "prajjwal1/bert-tiny"  # very small model suited for CPU (debugging, etc)
)

# What will we do?
# 1. Data
#  - Load the data from Hugging Face `datasets`
#  - Preprocess the data for our task
# 2. Model
#  - Load an AutoModelForSequenceClassification.from_pretrained
#  - Set up a suitable `LightningModule` for the `Trainer` which modularizes training, validation and testing
# 3. Evaluation
#  - Set up a suitable `LightningModule` for the `Trainer` which modularizes training, validation and testing
# 4. Modularize and CLI-fy our code with Hydra

# 1. Loading Data

# Documentation: https://huggingface.co/docs/datasets/loading
from datasets import load_dataset

# How to find suitable datasets?
# https://huggingface.co/datasets

# path: overarching dataset name
# name: "sub-dataset" (in MasakhanNER language)
training_data = load_dataset(path="glue", name="mnli", split="train")
validation_data = load_dataset(path="xnli", name="en", split="validation")
test_data = load_dataset(path="xnli", name="en", split="test")

# NLI as a task: https://cims.nyu.edu/~sbowman/multinli/

# How is a hypothesis to a premise? Natural language inference task
# Premise 	Label 	Hypothesis
# The Old One always comforted Ca'daan, except today. 	neutral 	Ca'daan knew the Old One very well.
# Your gift is appreciated by each and every student who will benefit from your generosity. 	neutral 	Hundreds of students will benefit from your generosity.
# yes now you know if if everybody like in August when everybody's on vacation or something we can dress a little more casual or 	contradiction 	August is a black out month for vacations in the company.
# At the other end of Pennsylvania Avenue, people began to line up for a White House tour. 	entailment 	People formed a line at the end of Pennsylvania Avenue.

# 2. Preprocessing Data


# HuggingFace `datasets.arrow_dataset.Dataset` supports fast batched preprocessing
# Be wary if your function is for an individual (dict[str, str | int | ...]) or a batch of examples (dict[str, list[str | int, ...])
def preprocess_nli(
    examples: dict[str, list[str | int]], # batched = True, dictionary of many (list of) indices
    tokenizer: PreTrainedTokenizerFast,
    tokenize_kwargs: dict[str, bool | int | str],
):
    batch = tokenizer(
        # casting just makes our type checker work more nicely
        text=cast(list[str], examples["premise"]),
        text_pair=cast(list[str], examples["hypothesis"]),
        **tokenize_kwargs,  # type: ignore
    )
    # # 
    # batch: dict[str, list[int]]
    return batch


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
training_data = training_data.map(
    function=preprocess_nli,
    batched=True, # 1000 instances at once
    fn_kwargs={
        "tokenizer": tokenizer,
        "tokenize_kwargs": {"max_length": 128, "truncation": True},
    },
)
validation_data = validation_data.map(
    function=preprocess_nli,
    batched=True,
    fn_kwargs={
        "tokenizer": tokenizer,
        "tokenize_kwargs": {"max_length": 128, "truncation": True},
    },
)
test_data = test_data.map(
    function=preprocess_nli,
    batched=True,
    fn_kwargs={
        "tokenizer": tokenizer,
        "tokenize_kwargs": {"max_length": 128, "truncation": True},
    },
)

# 3. DataLoading


def collate_fn(
    # list of dictionaries
    inputs: list[dict[str, list[int] | int]],
    tokenizer: PreTrainedTokenizerFast,
    padding_kwargs: dict = {"padding": True, "max_length": 128, "return_tensors": "pt"},
):
    features = [
        {
            "input_ids": line["input_ids"],
            "attention_mask": line["attention_mask"],
            "token_type_ids": line["token_type_ids"],
        }
        for line in inputs
    ]
    batch = tokenizer.pad(features, **padding_kwargs)
    batch["labels"] = torch.LongTensor([line["label"] for line in inputs])
    return batch


collator = partial(collate_fn, tokenizer=tokenizer)
# I now have my 390K dataset, and I want to sample individual batches of size=32
# DataLoader takes care of turning the dataset into an iterator that returns chunks of the dataset in the batch size
# Shuffling, collating, etc. taking care of
train_dataloader = DataLoader(
    training_data, collate_fn=collator, batch_size=32, shuffle=True
)
val_dataloader = DataLoader(
    validation_data, collate_fn=collator, batch_size=32, shuffle=False
)
test_dataloader = DataLoader(
    test_data, collate_fn=collator, batch_size=32, shuffle=False
)


class SequenceClassification(LightningModule):
    def __init__(self, model: nn.Module, lr: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self._validation_step_outputs = {"labels": [], "preds": []}
        self._test_step_outputs = {"labels": [], "preds": []}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, batch, *args, **kwargs):
        return self.model(**batch)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # outputs
        # classifier
        # compute cross entropy
        # Importantly, Huggingface models always try to compute a loss IF and ONLY IF you pass labels
        return self.model(**batch)

    def validation_step(self, batch, *args, **kwargs):
        outputs = self(batch)
        # NLC
        # C: number of classes
        # outputs.logits.argmax(-1) -> select the maximum over C
        # corresponds to prediction
        preds = outputs.logits.argmax(-1)
        self._validation_step_outputs["labels"].extend(batch["labels"])
        self._validation_step_outputs["preds"].extend(preds.tolist())

    def test_step(self, batch, *args, **kwargs):
        outputs = self(batch)
        preds = outputs.logits.argmax(-1)
        self._test_step_outputs["labels"].extend(batch["labels"])
        self._test_step_outputs["preds"].extend(preds.tolist())

    def on_validation_epoch_end(self) -> None:
        from torchmetrics.functional import accuracy

        acc = accuracy(
            preds=torch.LongTensor(self._validation_step_outputs["preds"]),
            target=torch.LongTensor(self._validation_step_outputs["labels"]),
            num_classes=3,
            task="multiclass",
        )
        self.log("val/acc", acc)
        self._validation_step_outputs = {"labels": [], "preds": []}

    def on_test_epoch_end(self) -> None:
        from torchmetrics.functional import accuracy

        acc = accuracy(
            preds=torch.LongTensor(self._test_step_outputs["preds"]),
            target=torch.LongTensor(self._test_step_outputs["labels"]),
            num_classes=3,
            task="multiclass",
        )
        self.log("test/acc", acc)
        self._test_step_outputs = {"labels": [], "preds": []}


model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path, num_labels=3
)
# we can have one ore more loggers
logger = CSVLogger(save_dir="./")
trainer = Trainer(accelerator="cpu", max_epochs=10, limit_train_batches=100)
module = SequenceClassification(model=model, lr=2e-5)
trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(module, dataloaders=test_dataloader)


from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
