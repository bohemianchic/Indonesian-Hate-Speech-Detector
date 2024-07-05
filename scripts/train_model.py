import pandas as pd
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from sklearn.model_selection import train_test_split
from custom_dataset import CustomDataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load prepared data
def load_prepared_data():
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    return train_data, test_data

# Load prepared data
logger.info("Loading prepared data...")
train_data, test_data = load_prepared_data()

# Tokenizer initialization
model_name = 'indolem/indobertweet-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize data
logger.info("Tokenizing data...")
train_encodings = tokenizer(list(train_data['Tweet']), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_data['Tweet']), truncation=True, padding=True, max_length=128)

# Create datasets
logger.info("Creating datasets...")
train_dataset = CustomDataset(train_encodings, list(train_data['Label']))
test_dataset = CustomDataset(test_encodings, list(test_data['Label']))

# Model initialization
logger.info("Initializing model...")
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Custom Trainer class with class weights
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Define class weights
class_weights = torch.tensor([0.6, 1.7])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.1,
    learning_rate=1e-6,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1
)

# Initialize the trainer
logger.info("Initializing the trainer...")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    class_weights=class_weights,  # Pass the class weights to the trainer
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Train the model
logger.info("Starting training...")
trainer.train()

# Evaluate the model
logger.info("Evaluating the model...")
results = trainer.evaluate()
logger.info(f"Results: {results}")
