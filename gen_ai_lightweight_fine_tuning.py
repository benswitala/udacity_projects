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




# SOLUTION






#trainer.train()



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
        output_dir="./sentiment_analysis",
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

trainer.evaluate()

from peft import LoraConfig, PeftModel, get_peft_model, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["distilbert.transformer.layer.0.attention.q_lin"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
    task_type=TaskType.SEQ_CLS

)






# Evaluate the PEFT model
# peft_trainer.evaluate()


model = get_peft_model(model, config)
model.print_trainable_parameters()

# Define new training arguments for the PEFT model
peft_training_args = TrainingArguments(
    output_dir="./peft_sentiment_analysis",
    learning_rate=2e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create a new Trainer instance for the PEFT model
peft_trainer = Trainer(
    model=model,
    args=peft_training_args,
    train_dataset= tokenized_ds['train'].rename_column('label', 'labels'),
    eval_dataset= tokenized_ds['test'].rename_column('label', 'labels'),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Train the PEFT model
peft_trainer.train()

# Save the PEFT model weights
model.save_pretrained("./peft_sentiment_analysis")

#!pip install --upgrade datasets==3.2.0 huggingface-hub==0.27.1

from peft import AutoPeftModelForSequenceClassification

fine_tuned_model = AutoPeftModelForSequenceClassification.from_pretrained("./peft_sentiment_analysis")




# Create the Trainer instance 
peft_training_args = TrainingArguments(
    output_dir="./peft_sentiment_analysis",
    learning_rate=2e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create a new Trainer instance for the PEFT model
peft_trainer = Trainer(
    model=fine_tuned_model,
    args=peft_training_args,
    train_dataset= tokenized_ds['train'].rename_column('label', 'labels'),
    eval_dataset= tokenized_ds['test'].rename_column('label', 'labels'),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Evaluate the model 
results = trainer.evaluate() 

# Print the evaluation results 
print(results)
