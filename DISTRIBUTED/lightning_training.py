import pytorch_lightning as L
import time
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
)

from tokenize_squad import TOKENIZED_PATH, get_tokenized_datasets

try:
    tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    print("Tokenized dataset found. Proceeding with training...")
except FileNotFoundError:
    print(
        f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
    )
    tokenized_datasets = get_tokenized_datasets()
    tokenized_datasets = load_from_disk(TOKENIZED_PATH)

bert_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")


class LanguageModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]

        outputs = self(input_ids, attention_mask, token_type_ids)
        start_loss = F.cross_entropy(outputs.start_logits, start_positions)
        end_loss = F.cross_entropy(outputs.end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]

        outputs = self(input_ids, attention_mask, token_type_ids)
        start_loss = F.cross_entropy(outputs.start_logits, start_positions)
        end_loss = F.cross_entropy(outputs.end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]

        outputs = self(input_ids, attention_mask, token_type_ids)
        start_loss = F.cross_entropy(outputs.start_logits, start_positions)
        end_loss = F.cross_entropy(outputs.end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        #return AdamW(self.parameters(), lr=2e-5)
        return torch.optim.SGD(self.parameters(), lr=0.01)


def main():
    L.seed_everything(42)

    try:
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)
        print("Tokenized dataset found. Proceeding with training...")
    except FileNotFoundError:
        print(
            f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
        )
        get_tokenized_datasets()
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)

    # Split data into train, val, test
    train_dataset = tokenized_datasets["train"].select(range(22500))
    val_dataset = tokenized_datasets["validation"].select(range(200))
    test_dataset = tokenized_datasets["validation"].select(range(5000, 5200))

    # Use DefaultDataCollator to handle conversion to tensors
    data_collator = DefaultDataCollator()

    train_dataloader = DataLoader(
        train_dataset, batch_size=150, shuffle=True, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator
    )

    # Model
    model = LanguageModel()

    # TensorBoard Logger
    logger = TensorBoardLogger("l_runs", name="bert_lightning")

    # PyTorch Profiler
    profiler = PyTorchProfiler(
        dirpath="l_runs/bert_lightning",
        filename="profiler",
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "l_runs/bert_lightning"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        #gradient_clip_val=0.25,
        max_epochs=6,
        logger=logger,  # Add the logger here
        profiler=profiler,  # Add the profiler here
        default_root_dir="finetune-l-BERT-squad"
    )
    start_time = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print("------------------------------------------------------------")
    print(f"Execution time: {execution_time:.4f} minutes")
    print("------------------------------------------------------------")
    trainer.test(model, test_dataloader)
    #Save model
    #trainer.save_checkpoint("finetune-l-BERT-squad/BERT_checkpoint.ckpt")

if __name__ == "__main__":
    main()
