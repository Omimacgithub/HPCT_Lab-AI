import time
import torch.cuda
from torch.utils.data import DataLoader
import torch.profiler
from datasets import load_from_disk
from tokenize_squad import TOKENIZED_PATH, get_tokenized_datasets
from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering, default_data_collator

try:
    tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    #['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']

    print("Tokenized dataset found. Proceeding with training...")
except FileNotFoundError:
    print(
        f"Tokenized dataset not found at {TOKENIZED_PATH}. Running tokenize-squad.py..."
    )
    tokenized_datasets = get_tokenized_datasets()

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

#SQUAD train dataset has 87599 rows
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=128, collate_fn=default_data_collator)
#For freeing GPU memory
del tokenized_datasets


args = TrainingArguments(
    "finetune-BERT-squad",
    #eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    #per_device_eval_batch_size=8,
    #num_train_epochs=1000,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=args,
    #train_dataset=tokenized_datasets["train"].select(range(1000)),
    #eval_dataset=tokenized_datasets["validation"].select(range(100)),
    #data_collator=data_collator,
    #tokenizer=tokenizer,
)

prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/BERTSQUAD'),
        record_shapes=True,
        with_stack=True)

num_epochs = 10 

start_time = time.time()
prof.start()
for epoch in range(num_epochs):
    #if(epoch == 0 or epoch == (num_epochs-1)):
    for step, batch_data in enumerate(train_dataloader):
            prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
            if step >= 1 + 1 + 3:
                break
            trainer.training_step(model, batch_data)
            #For freeing GPU memory
            torch.cuda.empty_cache()
prof.stop()
end_time = time.time()
execution_time = (end_time - start_time)/60
print(f"Execution time: {execution_time:.4f} minutes")

