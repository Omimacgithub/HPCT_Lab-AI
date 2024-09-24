#import torch
#import torch.tensor
from torch.utils.data import Dataset, DataLoader
import torch.profiler
#import torchvision.datasets
#import torchvision.transforms as T
from datasets import load_from_disk
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from tokenize_squad import TOKENIZED_PATH, get_tokenized_datasets

train_loader = None

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


try:
    tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    train_dataset = SquadDataset(tokenized_datasets["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    #['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
    # Load the dataset
    #squad = load_dataset("squad")

    # Load the tokenizer and model
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #train_tokens = tokenizer(squad["train"]["question"], padding = True, truncation=True)
    print("Tokenized dataset found. Proceeding with training...")
except FileNotFoundError:
    print(
        f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
    )
    tokenized_datasets = get_tokenized_datasets()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

data_collator = DefaultDataCollator()

args = TrainingArguments(
    "finetune-BERT-squad",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=125,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),
    eval_dataset=tokenized_datasets["validation"].select(range(100)),
    data_collator=data_collator,
    tokenizer=tokenizer,
)


prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/profiling'),
        record_shapes=True,
        with_stack=True)

prof.start()
for step, batch_data in enumerate(train_loader):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if step >= 1 + 1 + 3:
            break
        trainer.training_step(model, batch_data)
prof.stop()

