from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

import adabound

class FooNetwork(pl.LightningModule):
    def __init__(self, hparams):
        # init superclass
        super(FooNetwork, self).__init__()
        self.hparams = hparams

        self.n_channels = hparams.n_channels
        self.n_classes  = hparams.n_classes
        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        # TODO: implementation
        pass

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        # TODO: implementation

        return x

    def loss(self, y, p):
        loss = torch.nn.CrossEntropyLoss()(p, y)
        return loss

    def training_step(self, data_batch, batch_i):
        # forward pass
        x, y = data_batch
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        return output

    def validation_step(self, data_batch, batch_i):
        x, y = data_batch
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
        })

        return output

    def validation_end(self, outputs):
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dic = {'val_loss': val_loss_mean}

        return tqdm_dic

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def __dataloader(self, dataset):
        # when using multi-node we need to add the datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size  # scale batch size
        except Exception:
            pass

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=4
        )

        return loader

    @pl.data_loader
    def tng_dataloader(self):
        print('tng data loader called')
        # TODO: implementation
        dataset = None
        return self.__dataloader(dataset)

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        # TODO: implementation
        dataset = None
        return self.__dataloader(dataset)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(title='Network options')

        # network params
        parser.add_argument('--n_channels', default=1,  type=int,
                            help='input of model')
        parser.add_argument('--n_classes',  default=3,  type=int,
                            help='output of model')
        parser.add_argument('--batch_size', default=32, type=int,
                            help='minibatch size')
        parser.add_argument('--learning_rate', default=1e-3, type=float,
                            help='learning rate of optimizer')

        return parent_parser
