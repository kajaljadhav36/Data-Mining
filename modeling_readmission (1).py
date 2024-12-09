# coding=utf-8
import os
import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from torch.nn import LeakyReLU, PReLU

class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.activation = PReLU()  # Using PReLU for potential improvements in model performance

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_labels=2):
        config_file = os.path.join(pretrained_model_name_or_path, "bert_config.json")
        weights_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        config = BertConfig.from_json_file(config_file)
        model = cls(config, num_labels)
        model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        return model

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.activation(logits)  # Apply activation function

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
