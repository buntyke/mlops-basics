"""
============================================================================
Name        : lightning_module.py
Author      : Nishanth Koganti
Description : Lightning module to manage model training for classifier training
Source      : https://github.com/graviraja/MLOps-Basics 
============================================================================
"""

# import third party modules
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score


class ColaModel(pl.LightningModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', lr=1e-2):
        super(ColaModel, self).__init__()

        self.save_hyperparameters()

        self.num_classes = 2
        self.bert = AutoModel.from_pretrained(model_name)

        self.W = nn.Linear(
            self.bert.config.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(
            batch['input_ids'], batch['attention_mask'])

        loss = F.cross_entropy(logits, batch['label'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            batch['input_ids'], batch['attention_mask'])
        loss = F.cross_entropy(logits, batch['label'])
        
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch['label'].cpu())
        val_acc = torch.tensor(val_acc)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams['lr'])
