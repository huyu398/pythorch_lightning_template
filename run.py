import pathlib
import inspect

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

# example
from network import FooNetwork as Network

def parse_argdict_for_method(method, hparams):
    hparams_dict = vars(hparams)
    argument_dict = {}

    for argname in inspect.getfullargspec(method).args:
        if argname in hparams_dict:
            argument_dict[argname] = hparams_dict[argname]

    return argument_dict


def main(hparams):
    # init experiment
    experiment_args = parse_argdict_for_method(Experiment.__init__, hparams)
    exp = Experiment(
        **experiment_args
    )

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = Network(hparams)

    # callbacks
    if hparams.enable_early_stop:
        early_stop = EarlyStopping(
            monitor=hparams.monitor_value,
            patience=hparams.patience,
            verbose=True,
            mode=hparams.monitor_mode
        )
    else:
        early_stop = None

    if hparams.enable_model_checkpoint:
        model_save_path = pathlib.Path(exp.log_dir).parent / 'model_weights'
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=hparams.save_best_only,
            save_weights_only=hparams.save_weights_only,
            verbose=True,
            monitor=hparams.monitor_value,
            mode=hparams.monitor_mode
        )
    else:
        checkpoint = None

    # configure trainer
    trainer_args = parse_argdict_for_method(Trainer.__init__, hparams)
    trainer = Trainer(
        experiment=exp,
        early_stop_callback=early_stop,
        checkpoint_callback=checkpoint,
        **trainer_args
    )

    # train model
    trainer.fit(model)

import argparse

def default_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment parameters
    exp_parser = parser.add_argument_group(title='Experiment options')
    def save_dir(dir_path):
        dir_path = pathlib.Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError
        return dir_path
    exp_parser.add_argument('--save_dir', default=pathlib.Path(__file__).parent.absolute(), type=save_dir)
    exp_parser.add_argument('--name', default='default', type=str)
    exp_parser.add_argument('--debug', default=False, type=bool)
    exp_parser.add_argument('--version', default=None, type=int)
    exp_parser.add_argument('--autosave', default=False, type=bool)
    exp_parser.add_argument('--description', default=None, type=str)
    exp_parser.add_argument('--create_git_tag', default=False, type=bool)

    # EarlyStopping parameters
    estp_parser = parser.add_argument_group(title='EarlyStopping options')
    estp_parser.add_argument('--enable_early_stop', default=False, type=bool)
    estp_parser.add_argument('--patience', default=0, type=int)

    # ModelCheckpoint parameters
    chkpnt_parser = parser.add_argument_group(title='ModelCheckpoint options')
    chkpnt_parser.add_argument('--enable_model_checkpoint', default=True, type=bool)
    chkpnt_parser.add_argument('--save_best_only', default=False, type=bool)
    chkpnt_parser.add_argument('--save_weights_only', default=False, type=bool)

    # EarlyStopping and ModelCheckpoint parameters
    common_parser = parser.add_argument_group(title='EarlyStopping and ModelCheckpoint options')
    common_parser.add_argument('--monitor_value', default='val_loss', type=str)
    common_parser.add_argument('--monitor_mode', default='min', type=str,
                               choices=['auto', 'min', 'max'])

    # Trainer parameters
    trainer_parser = parser.add_argument_group(title='Trainer options')
    trainer_parser.add_argument('--max_nb_epochs', default=100, type=int, help='cap epochs')
    trainer_parser.add_argument('--min_nb_epochs', default=10, type=int, help='min epochs')
    trainer_parser.add_argument('--train_percent_check', default=1.0, type=float,
                                help='how much of train set to check')
    trainer_parser.add_argument('--val_percent_check', default=1.0, type=float,
                                help='how much of valid set to check')
    trainer_parser.add_argument('--test_percent_check', default=1.0, type=float,
                                help='how much of test set to check')
    trainer_parser.add_argument('--gpus', nargs='+', default=None, type=int,
                                help='set number of gpu to use')
    trainer_parser.add_argument('--use_amp', default=True, type=bool,
                                help='if true, use amp (need to install apex)')
    trainer_parser.add_argument('--amp_level', default='O2', type=str,
                                choices=['O0', 'O1', 'O2', 'O3'],
                                help='set amp optimization level')

    return parser

if __name__ == '__main__':
    # use default args given by lightning
    parent_parser = default_args()

    # allow model to overwrite or extend args
    parser = Network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)
