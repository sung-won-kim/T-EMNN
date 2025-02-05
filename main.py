import os
import time
import torch
import random
import pickle
import argparse
import importlib
import numpy as np
from utils import *
import lightning as L
from random import shuffle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader as PyGDataLoader
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor

def main(args):

    torch.manual_seed(args.seed) 
    random.seed(args.seed)       
    np.random.seed(args.seed)    

    module = importlib.import_module(f"model.{args.model}")
    T_EMNN = getattr(module, "T_EMNN")
    LargeDataset = getattr(module, "LargeDataset")

    # _________
    # Load Data
    train_data_list = os.listdir(f'./data/{args.data_fname}/train')
    valid_data_list = []
    test_data_list = os.listdir(f'./data/{args.data_fname}/test')

    train_data_list.sort()
    test_data_list.sort()

    train_list = []
    valid_list = []

    unique_shape_list = list(set([data.split('_')[0] for data in train_data_list]))
    unique_shape_list.sort()

    for shape in unique_shape_list:
        shape_data = [data for data in train_data_list if data.split('_')[0] == shape]
        shuffle(shape_data)

        if shape == unique_shape_list[-1]:
            valid_list = valid_list + shape_data
        else:
            divide_idx = int(len(shape_data) * args.train_rate)
            train_list = train_list + shape_data[:max(divide_idx,1)]
            valid_list = valid_list + shape_data[max(divide_idx,1):]

    train_data_list = train_list
    valid_data_list = valid_list
    
    class DataModule(L.LightningDataModule):
        def train_dataloader(self):
            train_dataset = LargeDataset(train_data_list,basepath=f'./data/{args.data_fname}/train', args=args)
            return PyGDataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            val_dataset = LargeDataset(valid_data_list,basepath=f'./data/{args.data_fname}/train', args=args)
            return PyGDataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

        def test_dataloader(self):
            test_dataset = LargeDataset(test_data_list,basepath=f'./data/{args.data_fname}/test', args=args)
            return PyGDataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
        
    # sample_data 
    with open(f'./data/{args.data_fname}/train/{train_data_list[0]}', 'rb') as f:
        sample_data = pickle.load(f)

    dirpath = f'./best_models/{args.summary}/s{args.seed}_{time.strftime("%m%d-%H%M")}/'

    checkpoint_callback = ModelCheckpoint(
        monitor='Valid RMSE',
        dirpath=dirpath,
        filename='best',
        save_top_k=1,
        save_last=True)

    if not os.path.exists(dirpath): 
        os.makedirs(dirpath) 

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5), LearningRateMonitor(logging_interval='epoch')],
        log_every_n_steps=1, 
        check_val_every_n_epoch=args.val_interval,
    )

    model = T_EMNN(sample_data, args)

    datamodule = DataModule()
        
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)

if __name__ == '__main__':

    timestr = time.strftime("%m$d")

    def list_of_ints(arg):
        if arg == 'cpu':
            return arg
        else:
            return list(map(int, arg.split(',')))
        
    def parse_args():
        parser = argparse.ArgumentParser()
        timestr = time.strftime("%m%d")

        parser.add_argument("--model", type=str, default='T_EMNN')
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--devices", type=list_of_ints, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--thres_lr", type=float, default=0.01)
        parser.add_argument("--sparsity_loss", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=5e-4)
        parser.add_argument("--train_rate", type=float, default=0.8)
        parser.add_argument("--val_interval", type=int, default=10)
        parser.add_argument("--summary", type=str, default=f'{timestr}')
        parser.add_argument("--data_fname", type=str, default='injection_mold')
        parser.add_argument("--alpha", type=float, default=3.0)
        return parser.parse_known_args()

    args, unknown = parse_args()

    main(args)
