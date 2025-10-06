
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# Load dataset and tokenizer
ds = load_dataset("dair-ai/emotion")
tokenizer = AutoTokenizer.from_pretrained("./distilled-distilbert-3layer")
model = AutoModelForSequenceClassification.from_pretrained("./distilled-distilbert-3layer")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

# Tokenize splits
train = ds["train"].map(tokenize, batched=True)
valid = ds["validation"].map(tokenize, batched=True)
cols = ["input_ids", "attention_mask", "label"]
train.set_format("torch", columns=cols)
valid.set_format("torch", columns=cols)

args = TrainingArguments(
    output_dir="./distilled-3layer-ft",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    num_train_epochs=3,
    eval_strategy="epoch",   # corrected arg name
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    train_dataset=train,
    eval_dataset=valid,
)

trainer.train()

