#this is partially copied from a Udacity exercise, and I liberally used LLMs (Udacity AI, Bing CoPilot, Google Gemini) to write the rest.

#Dear Student

#Please use AutoModelForSequenceClassification for loading the pretrained model

#And AutoPeftModelforSequenceClassification for reloading the fine tuned model

###############################################################################
# Import the datasets and transformers packages

from datasets import load_dataset

# Load the train and test splits of the imdb dataset
splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("imdb", split=splits))}

# Thin out the dataset to make it run faster for this example
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

# Show the dataset
ds

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)


# Check that we tokenized the examples properly
assert tokenized_ds["train"][0]["input_ids"][:5] == [101, 2045, 2003, 2053, 7189]

# Show the first example of the tokenized training set
print(tokenized_ds["train"][0]["input_ids"])


# SOLUTION

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.base_model.parameters():
    param.requires_grad = False

model.classifier

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

from peft import LoraConfig, PeftModel

# Define LoraConfig
# Define LoraConfig (without 'alpha')
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    target_modules=["classifier"],  # Apply Lora to the classifier layer
)

# Create the PeftModel
peft_model = PeftModel.from_pretrained(model)

# Freeze base model parameters
for param in peft_model.base_model.parameters():
    param.requires_grad = False

# Print the number of trainable parameters
print(sum(p.numel() for p in peft_model.parameters() if p.requires_grad)) 
