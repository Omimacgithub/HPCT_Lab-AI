from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

data_collator = DefaultDataCollator()

args = TrainingArguments(
    "finetune-BERT-squad",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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

trainer.train()

