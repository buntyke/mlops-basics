"""
============================================================================
Name        : train.py
Author      : Nishanth Koganti
Description : Main script to train the classifier
Source      : https://github.com/graviraja/MLOps-Basics 
============================================================================
"""

# import modules
import logging
import argparse

# import third party modules
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# relative imports
from data_module import DataModule
from train_module import ColaModel


def main():
    parser = argparse.ArgumentParser(
        description='Text classifier script')
    parser.add_argument(
        '-m', '--models', type=str, default='models',
        help='Path to models directory')
    parser.add_argument(
        '-l', '--logs', type=str, default='logs',
        help='Path to logs directory')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.DEBUG)

    # setup the nlp dataset
    cola_dataset = load_dataset('glue', 'cola')
    logging.info('Dataset Info: ')
    logging.info(str(cola_dataset))

    # view sample from the dataset
    train_dataset = cola_dataset['train']
    logging.info('Dataset Sample: ')
    logging.info(str(train_dataset[0]))

    # setup cola datamodule and cola model classes
    cola_data = DataModule()
    cola_model = ColaModel()

    # setup the callback functions
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.models, monitor='val_loss', mode='min')
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=3, verbose=True, mode='min')

    # setup the logger
    cola_logger = pl.loggers.TensorBoardLogger(
        'logs/', name='cola', version=1)

    # setup the model trainer
    trainer = pl.Trainer(
        default_root_dir=args.logs,
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=10, fast_dev_run=False, logger=cola_logger,
        callbacks=[checkpoint_callback, early_stopping_callback])

    # train the cola model
    trainer.fit(cola_model, cola_data)


if __name__=='__main__':
    main()
