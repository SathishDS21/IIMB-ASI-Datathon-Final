import pandas as pd
import torch
import os
import numpy as np
import pickle
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up device for GPU/CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_training_data(file_path):
    logging.info(f"Loading training data from: {file_path}")
    df = pd.read_excel(file_path)
    if 'Keyword' not in df or 'Category' not in df:
        raise ValueError("Input file must contain 'Keyword' and 'Category' columns.")
    df = df[['Keyword', 'Category']].dropna()
    logging.info(f"Loaded {len(df)} rows.")
    return df


def preprocess_data(df):
    logging.info("Preprocessing data and encoding labels.")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Category'])
    logging.info(f"Classes found: {list(label_encoder.classes_)}")
    return df, label_encoder


def check_class_distribution(df):
    class_counts = df['label'].value_counts()
    logging.info("Class distribution:\n%s", class_counts)
    return class_counts


def handle_small_classes(df):
    class_counts = check_class_distribution(df)
    small_classes = class_counts[class_counts < 2].index.tolist()

    for class_label in small_classes:
        class_df = df[df['label'] == class_label]
        df = pd.concat([df, class_df])

    return df


def tokenize_function(tokenizer, texts, labels):
    logging.info("Tokenizing text data using T5 tokenizer.")

    tokenized_data = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    labels = [str(label) for label in labels]
    tokenized_data['labels'] = tokenizer(
        labels,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt"
    )['input_ids']

    return tokenized_data


def compute_metrics(eval_pred):
    # Extract predictions and labels from EvalPrediction
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # If logits is a tuple, extract the first element (the logits themselves)
    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert logits to tensor if it's a NumPy array
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    # Get the token predictions from logits (select the token with the highest probability)
    pred_ids = torch.argmax(logits, dim=-1)

    # Decode the predictions and labels as text
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compare the generated text directly with the expected text
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Calculate accuracy by comparing the predicted text with the label text
    accuracy = accuracy_score(decoded_labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(decoded_labels, decoded_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_model(train_dataset, val_dataset, tokenizer, label_encoder, class_weights, model_name='t5-small'):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        gradient_accumulation_steps=2
    )

    if device.type == "mps":
        training_args.fp16 = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()

    output_dir = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM Model"
    logging.info(f"Saving model and tokenizer to {output_dir}.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    logging.info("Training complete and models saved successfully.")


if __name__ == "__main__":
    training_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Training Data/Training data.xlsx"

    df = load_training_data(training_file_path)
    df, label_encoder = preprocess_data(df)

    # Handle small classes by duplicating rows with fewer than 2 samples
    df = handle_small_classes(df)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Initialize T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Tokenize the training and validation data
    tokenized_train = tokenize_function(tokenizer, train_df['Keyword'], train_df['label'])
    tokenized_val = tokenize_function(tokenizer, val_df['Keyword'], val_df['label'])

    # Convert tokenized data into HuggingFace-style datasets
    train_dataset = Dataset.from_dict(tokenized_train)
    val_dataset = Dataset.from_dict(tokenized_val)

    class_weights = torch.ones(len(label_encoder.classes_)).to(device)

    # Train the model
    train_model(train_dataset, val_dataset, tokenizer, label_encoder, class_weights)