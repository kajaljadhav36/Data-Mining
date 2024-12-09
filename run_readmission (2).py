import os
import logging
import argparse
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from torch.optim import AdamW
from tqdm import trange
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
log_manager = logging.getLogger(__name__)

# Data Classes
class DataSample:
    def __init__(self, identifier, primary_text, secondary_text=None, target_label=None):
        self.identifier = identifier
        self.primary_text = primary_text
        self.secondary_text = secondary_text
        self.target_label = target_label

class DataFeatures:
    def __init__(self, token_ids, attention_mask, label_index):
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.label_index = label_index

class DatasetProcessor:
    @classmethod
    def read_csv_file(cls, file_path):
        data_frame = pd.read_csv(file_path)
        expected_columns = ["ID", "TEXT", "Label"]
        if not all(column in data_frame.columns for column in expected_columns):
            raise ValueError(f"The CSV file {file_path} must contain these columns: {expected_columns}")
        return zip(data_frame.ID, data_frame.TEXT, data_frame.Label)

class AdmissionProcessor(DatasetProcessor):
    def load_training_data(self, directory_path):
        log_manager.info(f"Reading training data from {os.path.join(directory_path, 'train.csv')}")
        return self.create_samples(self.read_csv_file(os.path.join(directory_path, "train.csv")), "train")

    def load_validation_data(self, directory_path):
        log_manager.info(f"Reading validation data from {os.path.join(directory_path, 'val.csv')}")
        return self.create_samples(self.read_csv_file(os.path.join(directory_path, "val.csv")), "validation")

    def load_test_data(self, directory_path):
        log_manager.info(f"Reading test data from {os.path.join(directory_path, 'test.csv')}")
        return self.create_samples(self.read_csv_file(os.path.join(directory_path, "test.csv")), "test")

    def get_class_labels(self):
        return ["0", "1"]

    def create_samples(self, data_lines, dataset_type):
        samples = []
        for index, line in enumerate(data_lines):
            identifier = f"{dataset_type}-{index}"
            primary_text = line[1]
            target_label = int(line[2])
            samples.append(DataSample(identifier=identifier, primary_text=primary_text, target_label=target_label))
        return samples

# Convert Samples to Features
def convert_samples_to_features(samples, tokenizer, max_length):
    feature_list = []
    for sample in samples:
        encoded_data = tokenizer(
            sample.primary_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        token_ids = encoded_data["input_ids"].squeeze(0)
        attention_mask = encoded_data["attention_mask"].squeeze(0)
        label_index = int(sample.target_label)

        feature_list.append(DataFeatures(token_ids=token_ids, attention_mask=attention_mask, label_index=label_index))
    return feature_list

# Evaluate and Generate ROC AUC
def generate_roc_curve_and_metrics(model, data_loader, device, output_file):
    model.eval()
    predictions = []
    actual_labels = []
    probabilities = []

    with torch.no_grad():
        for data_batch in data_loader:
            token_ids, masks, true_labels = [data.to(device) for data in data_batch]

            model_outputs = model(input_ids=token_ids, attention_mask=masks)
            prediction_scores = model_outputs.logits
            positive_probs = torch.softmax(prediction_scores, dim=1)[:, 1]

            batch_predictions = torch.argmax(prediction_scores, dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
            actual_labels.extend(true_labels.cpu().numpy())
            probabilities.extend(positive_probs.cpu().numpy())

    # Calculate ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(actual_labels, probabilities)
    roc_auc_value = auc(false_positive_rate, true_positive_rate)

    # Plot ROC Curve
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color="orange", lw=2, label=f"ROC Curve (AUC = {roc_auc_value:.2f})")
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Analysis")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(output_file)
    plt.close()

    # Generate Classification Metrics
    classification_metrics = classification_report(actual_labels, predictions, target_names=["Class 0", "Class 1"])
    accuracy_score = np.mean(np.array(predictions) == np.array(actual_labels))

    return roc_auc_value, classification_metrics, accuracy_score

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", required=True)
    parser.add_argument("--pretrained_model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--max_token_length", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--training_epochs", default=10, type=int)
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--evaluate_model", action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_manager.info(f"Using device: {device}")

    # Set Random Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    processor = AdmissionProcessor()
    tokenizer = BertTokenizer.from_pretrained(arguments.pretrained_model)
    model = BertForSequenceClassification.from_pretrained(arguments.pretrained_model, num_labels=2)
    model.to(device)

    # Training Phase
    if arguments.train_model:
        training_samples = processor.load_training_data(arguments.data_directory)
        training_features = convert_samples_to_features(training_samples, tokenizer, arguments.max_token_length)

        train_dataset = TensorDataset(
            torch.stack([feature.token_ids for feature in training_features]),
            torch.stack([feature.attention_mask for feature in training_features]),
            torch.tensor([feature.label_index for feature in training_features], dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=arguments.batch_size)

        optimizer = AdamW(model.parameters(), lr=arguments.learning_rate)
        total_training_steps = len(train_loader) * arguments.training_epochs
        learning_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_training_steps
        )

        model.train()
        for epoch in trange(arguments.training_epochs, desc="Training Epochs"):
            epoch_loss = 0
            for data_batch in train_loader:
                token_ids, masks, labels = tuple(data.to(device) for data in data_batch)

                model_outputs = model(input_ids=token_ids, attention_mask=masks, labels=labels)
                loss_value = model_outputs.loss
                epoch_loss += loss_value.item()

                loss_value.backward()
                optimizer.step()
                learning_scheduler.step()
                optimizer.zero_grad()

            log_manager.info(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(train_loader):.4f}")

        # Save Model
        os.makedirs(arguments.output_directory, exist_ok=True)
        model.save_pretrained(arguments.output_directory)
        tokenizer.save_pretrained(arguments.output_directory)

    # Evaluation Phase
    if arguments.evaluate_model:
        validation_samples = processor.load_validation_data(arguments.data_directory)
        validation_features = convert_samples_to_features(validation_samples, tokenizer, arguments.max_token_length)

        validation_dataset = TensorDataset(
            torch.stack([feature.token_ids for feature in validation_features]),
            torch.stack([feature.attention_mask for feature in validation_features]),
            torch.tensor([feature.label_index for feature in validation_features], dtype=torch.long),
        )
        validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=arguments.batch_size)

        roc_auc, metrics, accuracy = generate_roc_curve_and_metrics(
            model, validation_loader, device, output_file="validation_roc_curve.png"
        )
        log_manager.info(f"Validation AUC: {roc_auc:.4f}")
        log_manager.info(f"Validation Metrics:\n{metrics}")
        log_manager.info(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
