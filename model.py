# Required Imports
import os
import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from torch.nn import LeakyReLU, PReLU

# Define the Custom BERT Model for Sequence Classification
class BertSequenceClassifier(nn.Module):
    def __init__(self, config, num_labels):
        """
        Initializes the model by defining its architecture.
        
        Args:
            config (BertConfig): Configuration object for the BERT model.
            num_labels (int): The number of possible output labels for classification.
        """
        super(BertSequenceClassifier, self).__init__()
        self.class_count = num_labels
        self.bert_model = BertModel(config)  # Load BERT base model
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)  # Dropout layer for regularization
        self.fc_layer = nn.Linear(config.hidden_size, num_labels)  # Linear layer for classification
        self.activation_function = PReLU()  # Parametric ReLU (PReLU) activation function for potential performance improvements

    @classmethod
    def load_model_from_path(cls, pretrained_model_name_or_path, num_labels=2):
        """
        Load a pre-trained BERT model from the given path or model name.
        
        Args:
            pretrained_model_name_or_path (str): Path to pre-trained model directory or model name.
            num_labels (int): Number of labels for classification (default is 2 for binary classification).
        
        Returns:
            BertSequenceClassifier: A model instance with loaded weights.
        """
        # Load the BERT configuration and weights from the specified path
        config_file = os.path.join(pretrained_model_name_or_path, "bert_config.json")
        weights_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        
        # Load configuration and initialize the model
        config = BertConfig.from_json_file(config_file)
        model = cls(config, num_labels)
        
        # Load pre-trained weights into the model
        model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        return model

    def pass_forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Defines the pass_forward pass for the model during both training and inference.
        
        Args:
            input_ids (torch.Tensor): Input token IDs (tokenized text).
            token_type_ids (torch.Tensor, optional): Token type IDs for distinguishing different parts of the input.
            attention_mask (torch.Tensor, optional): Attention mask to ignore padding tokens.
            labels (torch.Tensor, optional): Labels for training (used to calculate loss).
        
        Returns:
            torch.Tensor or tuple: Model output logits, or loss and logits during training.
        """
        # Get the pooled output from the BERT model (a representation of the entire input sequence)
        _, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        # Apply dropout for regularization
        pooled_output = self.dropout_layer(pooled_output)
        
        # Classify the pooled output into the given number of labels
        logits = self.fc_layer(pooled_output)
        
        # Apply the PReLU activation function
        logits = self.activation_function(logits)
        
        # If labels are provided, calculate and return the loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.class_count), labels.view(-1))
            return loss, logits
        
        # Return logits during inference (no labels provided)
        return logits
