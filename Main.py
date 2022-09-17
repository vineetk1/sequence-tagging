'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from Data import Data
from Model import Model
from ast import literal_eval
from sys import argv
import collections.abc
from pathlib import Path
from yaml import dump, full_load
from typing import Dict
from logging import getLogger
from utils.log_configuration import LOG_CONFIG
from logging.config import dictConfig

logg = getLogger(__name__)
dictConfig(LOG_CONFIG)


def main():
    # last file name in command-line has dictionaries of parameters
    params_file_path = argv[len(argv) - 1]
    # get user-provided-parameters
    with open(params_file_path, 'r') as paramF:
        user_dicts = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]
    user_dicts_keys = [
        'misc', 'optz_sched', 'data', 'trainer', 'model_init',
        'ld_resume_chkpt'
    ]
    if len(user_dicts) != len(user_dicts_keys):
        strng = (f'{argv[1]} MUST have {len(user_dicts_keys)} '
                 f'dictionaries even if the dictionaries are empty.')
        logg.critical(strng)
        exit()
    user_dicts = {k: v for k, v in zip(user_dicts_keys, user_dicts)}
    verify_and_change_user_provided_parameters(user_dicts)

    seed_everything(63)

    # change user-provided-parameters based on whether loading-from-checkpoint,
    # or resuming-from-checkpoint, or starting from scratch
    if 'ld_chkpt' in user_dicts['ld_resume_chkpt']:
        dirPath = Path(user_dicts['ld_resume_chkpt']['ld_chkpt']).resolve(
            strict=True).parents[1]
        chkpt_dicts = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        assert len(user_dicts) == len(chkpt_dicts)
        # override  certain user_dicts with chkpt_dicts; also if
        # user_dicts[user_dict_k] is empty then replace its content by
        # corresponding chkpt_dicts
        for user_dict_k in user_dicts_keys:
            if ((not user_dicts[user_dict_k]) and
                (user_dict_k != 'ld_resume_chkpt') and
                (user_dict_k != 'misc') and
                (user_dict_k != 'optz_sched')) or (user_dict_k
                                                   == 'model_init'):
                user_dicts[user_dict_k] = chkpt_dicts[user_dict_k]
    elif 'resume_from_checkpoint' in user_dicts['ld_resume_chkpt']:
        dirPath = Path(
            user_dicts['ld_resume_chkpt']['resume_from_checkpoint']).resolve(
                strict=True).parents[1]
        chkpt_dicts = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        assert len(user_dicts) == len(chkpt_dicts)
        # override  certain user_dicts with chkpt_dicts; also if
        # user_dicts[user_dict_k] is empty then replace its content by
        # corresponding chkpt_dicts; NOTE that the assumption is that
        # model_init and optz_sched values in chkpt_dicts are same as those in
        # checkpoint file
        for user_dict_k in user_dicts_keys:
            if ((not user_dicts[user_dict_k]) and
                (user_dict_k != 'ld_resume_chkpt') and
                (user_dict_k != 'misc')) or (user_dict_k == 'model_init') or (
                    user_dict_k == 'optz_sched'):
                user_dicts[user_dict_k] = chkpt_dicts[user_dict_k]
    else:
        tb_subDir = ",".join([
            f'{item}={(user_dicts["model_init"][item]).replace("/", "_")}'
            for item in ['model', 'model_type', 'tokenizer_type']
            if item in user_dicts['model_init']
        ])
        dirPath = Path('tensorboard_logs').joinpath(tb_subDir).resolve(
            strict=False)
        dirPath.mkdir(parents=True, exist_ok=True)

    # prepare and split dataset
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(
        user_dicts['model_init']['model_type'])
    data = Data(tokenizer=tokenizer,
                batch_size=user_dicts['data']['batch_size']
                if 'batch_size' in user_dicts['data'] else {})
    data.generate_data_labels(dataset_path=user_dicts['data']['dataset_path'])
    dataset_metadata = data.split_dataset(
        dataset_path=user_dicts['data']['dataset_path'],
        dataset_split=user_dicts['data']['dataset_split']
        if 'dataset_split' in user_dicts['data'] else {},
        train=user_dicts['misc']['train'],
        predict=user_dicts['misc']['predict'])

    # initialize model
    if 'ld_chkpt' in user_dicts['ld_resume_chkpt']:
        model = Model.load_from_checkpoint(
            checkpoint_path=user_dicts['ld_resume_chkpt']['ld_chkpt'])
    else:
        model = Model(model_init=user_dicts['model_init'],
                      tokenLabels_NumberCount=dataset_metadata[
                          'train token-labels -> number:count'])
    # batch_size is only provided to turn-off Lightning Warning;
    # resume_from_checkpoint can provide a different batch_size which will
    # conflict with this batch_size
    model.params(optz_sched_params=user_dicts['optz_sched'],
                 batch_size=dataset_metadata['batch size'])

    # create a directory to store all types of results
    if 'resume_from_checkpoint' in user_dicts['ld_resume_chkpt']:
        tb_logger = TensorBoardLogger(save_dir=dirPath.parent,
                                      name="",
                                      version=dirPath.name)
    else:
        new_version_num = max((int(dir.name.replace('version_', ''))
                               for dir in dirPath.glob('version_*')),
                              default=-1) + 1
        tb_logger = TensorBoardLogger(save_dir=dirPath,
                                      name="",
                                      version=new_version_num)
        dirPath = dirPath.joinpath('version_' + f'{new_version_num}')
        dirPath.mkdir(parents=True, exist_ok=True)
    paramFile = dirPath.joinpath('hyperparameters_used.yaml')
    paramFile.touch()
    paramFile.write_text(dump(user_dicts))

    # setup Callbacks and Trainer
    if user_dicts['misc']['train']:
        # (train, val, test): True
        ckpt_filename = ""
        for item in user_dicts['optz_sched']:
            if isinstance(user_dicts['optz_sched'][item], str):
                ckpt_filename += f'{item}={user_dicts["optz_sched"][item]},'
            elif isinstance(user_dicts['optz_sched'][item],
                            collections.abc.Iterable):
                for k, v in user_dicts['optz_sched'][item].items():
                    ckpt_filename += f'{k}={v},'
        ckpt_filename += '{epoch:02d}-{val_loss:.5f}'

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=user_dicts['misc']['save_top_k']
            if 'save_top_k' in user_dicts['misc'] else 1,
            save_last=True,
            every_n_epochs=1,
            filename=ckpt_filename)
        lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                         log_momentum=True)
        trainer = Trainer(logger=tb_logger,
                          deterministic=True,
                          num_sanity_val_steps=0,
                          callbacks=[
                              checkpoint_callback, lr_monitor,
                              TQDMProgressBar(refresh_rate=10)
                          ],
                          **user_dicts['trainer'])
    elif user_dicts['misc']['predict']:
        trainer = Trainer(logger=tb_logger,
                          num_sanity_val_steps=0,
                          log_every_n_steps=100,
                          enable_checkpointing=False,
                          **user_dicts['trainer'])
    else:
        # (train, val, test): False, predict: False
        strng = ('User specified train=False and predict=False. Must '
                 'train or predict or both.')
        logg.critical(strng)
        exit()

    # training and testing
    trainer.tune(model, datamodule=data)
    if user_dicts['misc']['train']:
        # (train, val, test): True
        trainer.fit(
            model=model,
            ckpt_path=user_dicts['ld_resume_chkpt']['resume_from_checkpoint']
            if 'resume_from_checkpoint' in user_dicts['ld_resume_chkpt'] else
            None,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader())
        # for testing, auto-load checkpoint file with lowest val-loss
        trainer.test(dataloaders=data.test_dataloader(), ckpt_path='best')
    if user_dicts['misc']['predict']:
        model.prepare_for_predict(dataset_meta=dataset_metadata,
                                  dirPath=dirPath,
                                  tokenizer=tokenizer)
        if user_dicts['misc']['train']:
            trainer.predict(dataloaders=data.predict_dataloader(),
                            ckpt_path='best')
        else:
            trainer.predict(model=model, dataloaders=data.predict_dataloader())
    logg.info(f"Results and other information is at the directory: {dirPath}")


def verify_and_change_user_provided_parameters(user_dicts: Dict):
    if 'ld_chkpt' in user_dicts[
            'ld_resume_chkpt'] and 'resume_chkpt' in user_dicts[
                'ld_resume_chkpt']:
        logg.critical('Cannot load- and resume-checkpoint at the same time')
        exit()

    if 'resume_from_checkpoint' in user_dicts[
            'ld_resume_chkpt'] and 'resume_from_checkpoint' in user_dicts[
                'trainer']:
        strng = (f'Remove "resume_from_checkpoint" from the "trainer" '
                 f'dictionary in the file {argv[1]}.')
        logg.critical(strng)
        exit()

    for k in ('train', 'predict'):
        if k in user_dicts['misc']:
            if not isinstance(user_dicts['misc'][k], bool):
                strng = (
                    f'value of "{k}" must be a boolean in misc dictionary '
                    f'of file {argv[1]}.')
                logg.critical(strng)
                exit()
        else:
            user_dicts['misc'][k] = False

    if not user_dicts['misc']['train'] and user_dicts['misc'][
            'predict'] and 'ld_chkpt' not in user_dicts['ld_resume_chkpt']:
        strng = ('Path to a checkpoint file must be specified if  '
                 'train=False and predict=True.')
        logg.critical(strng)
        exit()

    if not user_dicts['ld_resume_chkpt']:
        if user_dicts["model_init"][
                'model'] != "bert" or user_dicts["model_init"][
                    'tokenizer_type'] != "bert" or user_dicts[
                        "model_init"]['model_type'] != "bert-large-uncased":
            strng = ('unknown model and tokenizer_type: '
                     f'{user_dicts["model_init"]["model"]}'
                     f'{user_dicts["model_init"]["model_type"]}'
                     f'{user_dicts["model_init"]["tokenizer_type"]}')
            logg.critical(strng)
            exit()

        if not ('dataset_path' in user_dicts['data']
                and isinstance(user_dicts['data']['dataset_path'], str)):
            logg.critical('Must specify a path to the dataset.')
            exit()


if __name__ == '__main__':
    main()
