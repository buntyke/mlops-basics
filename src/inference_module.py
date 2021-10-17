"""
============================================================================
Name        : inference_module.py
Author      : Nishanth Koganti
Description : Inference module for predictions on test data
Source      : https://github.com/graviraja/MLOps-Basics 
============================================================================
"""

# import third party modules
import torch

# relative imports
from data_module import DataModule
from train_module import ColaModel


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path

        # loading the trained model
        self.model = ColaModel.load_from_checkpoint(model_path)

        # keep the model in eval mode
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ['unacceptable', 'acceptable']

    def predict(self, text):
        # text => run time input
        inference_sample = {'sentence': text}

        # tokenizing the input
        processed = self.processor.tokenize_data(inference_sample)

        # predictions
        logits = self.model(
            torch.tensor([processed['input_ids']]),
            torch.tensor([processed['attention_mask']]))

        scores = self.softmax(logits[0]).tolist()

        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({'label': label, 'score': score})

        return predictions
