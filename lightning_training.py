import pytorch_lightning as L
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tokenize_squad import TOKENIZED_PATH, get_tokenized_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
)

try:
    tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    print("Tokenized dataset found. Proceeding with training...")
except FileNotFoundError:
    print(
        f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
    )
    tokenized_datasets = get_tokenized_datasets()

bert_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")


class LanguageModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = bert_model

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def main():
    L.seed_everything(42)

    try:
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)
        # ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']

        print("Tokenized dataset found. Proceeding with training...")
    except FileNotFoundError:
        print(
            f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
        )
        get_tokenized_datasets()
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)

    # Split data in to train, val, test
    train_dataset = tokenized_datasets["train"].select(range(20000))
    val_dataset = tokenized_datasets["validation"].select(range(2000))
    test_dataset = tokenized_datasets["validation"].select(range(5000, 6000))
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Model
    model = LanguageModel()

    # Trainer
    trainer = L.Trainer(accelerator="gpu", gradient_clip_val=0.25, max_epochs=1)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
