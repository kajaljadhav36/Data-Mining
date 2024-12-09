'''
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
logger = logging.getLogger(__name__)

# Processor Classes
class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures:
    def __init__(self, input_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

class DataProcessor:
    @classmethod
    def _read_csv(cls, input_file):
        file = pd.read_csv(input_file)
        required_columns = ["ID", "TEXT", "Label"]
        if not all(col in file.columns for col in required_columns):
            raise ValueError(f"The CSV file {input_file} must contain the following columns: {required_columns}")
        return zip(file.ID, file.TEXT, file.Label)

class ReadmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_eval_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'val.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "eval")

    def get_test_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'test.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[1]
            label = int(line[2])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

# Convert Examples to Features
def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for example in examples:
        inputs = tokenizer(
            example.text_a,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label_id = int(example.label)

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label_id=label_id))
    return features

# Evaluate Model and Plot ROC AUC
def evaluate_and_plot_roc(model, dataloader, device, output_path):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probability

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    return roc_auc

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    processor = ReadmissionProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)

        train_data = TensorDataset(
            torch.stack([f.input_ids for f in train_features]),
            torch.stack([f.attention_mask for f in train_features]),
            torch.tensor([f.label_id for f in train_features], dtype=torch.long),
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        model.train()
        for epoch in trange(args.num_train_epochs, desc="Epoch"):
            total_loss = 0
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss}")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        # Validation Evaluation
        logger.info("Starting Validation evaluation...")
        eval_examples = processor.get_eval_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length)

        eval_data = TensorDataset(
            torch.stack([f.input_ids for f in eval_features]),
            torch.stack([f.attention_mask for f in eval_features]),
            torch.tensor([f.label_id for f in eval_features], dtype=torch.long),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

        evaluate_and_plot_roc(model, eval_dataloader, device, output_path="validation_roc_curve.png")
        logger.info("Validation evaluation complete.")

        # Test Evaluation
        logger.info("Starting Test evaluation...")
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)

        test_data = TensorDataset(
            torch.stack([f.input_ids for f in test_features]),
            torch.stack([f.attention_mask for f in test_features]),
            torch.tensor([f.label_id for f in test_features], dtype=torch.long),
        )
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)

        test_roc_auc = evaluate_and_plot_roc(model, test_dataloader, device, output_path="test_roc_curve.png")
        logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")

if __name__ == "__main__":
    main()
'''

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
logger = logging.getLogger(__name__)

# Processor Classes
class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures:
    def __init__(self, input_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

class DataProcessor:
    @classmethod
    def _read_csv(cls, input_file):
        file = pd.read_csv(input_file)
        required_columns = ["ID", "TEXT", "Label"]
        if not all(col in file.columns for col in required_columns):
            raise ValueError(f"The CSV file {input_file} must contain the following columns: {required_columns}")
        return zip(file.ID, file.TEXT, file.Label)

class ReadmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_eval_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'val.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "eval")

    def get_test_examples(self, data_dir):
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'test.csv')}")
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[1]
            label = int(line[2])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

# Convert Examples to Features
def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for example in examples:
        inputs = tokenizer(
            example.text_a,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label_id = int(example.label)

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label_id=label_id))
    return features

# Evaluate Model and Plot ROC AUC
def evaluate_and_plot_roc(model, dataloader, device, output_path):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probability

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    # Classification Report and Accuracy
    report = classification_report(all_labels, all_preds, target_names=["Not Readmitted", "Readmitted"])
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return roc_auc, report, accuracy

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    processor = ReadmissionProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)

        train_data = TensorDataset(
            torch.stack([f.input_ids for f in train_features]),
            torch.stack([f.attention_mask for f in train_features]),
            torch.tensor([f.label_id for f in train_features], dtype=torch.long),
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        model.train()
        for epoch in trange(args.num_train_epochs, desc="Epoch"):
            total_loss = 0
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss}")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        # Validation Evaluation
        logger.info("Starting Validation evaluation...")
        eval_examples = processor.get_eval_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length)

        eval_data = TensorDataset(
            torch.stack([f.input_ids for f in eval_features]),
            torch.stack([f.attention_mask for f in eval_features]),
            torch.tensor([f.label_id for f in eval_features], dtype=torch.long),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

        evaluate_and_plot_roc(model, eval_dataloader, device, output_path="validation_roc_curve.png")
        logger.info("Validation evaluation complete.")

        # Test Evaluation
        logger.info("Starting Test evaluation...")
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)

        test_data = TensorDataset(
            torch.stack([f.input_ids for f in test_features]),
            torch.stack([f.attention_mask for f in test_features]),
            torch.tensor([f.label_id for f in test_features], dtype=torch.long),
        )
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)

        test_roc_auc, test_report, test_accuracy = evaluate_and_plot_roc(
            model, test_dataloader, device, output_path="test_roc_curve.png"
        )
        logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info("Test Classification Report:\n" + test_report)

if __name__ == "__main__":
    main()
