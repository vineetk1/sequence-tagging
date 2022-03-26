'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from Data import Data
from Model import Model
from ast import literal_eval
from sys import argv
import collections.abc
from pathlib import Path
from yaml import dump, full_load
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
    if 'ld_chkpt' in user_dicts[
            'ld_resume_chkpt'] and 'resume_chkpt' in user_dicts[
                'ld_resume_chkpt']:
        logg.critical('Cannot load- and resume-checkpoint at the same time')
        exit()

    seed_everything(63)

    # change user-provided-parameters based on whether loading-from-checkpoint,
    # or resuming-from-checkpoint, or starting from scratch
    if 'ld_chkpt' in user_dicts['ld_resume_chkpt']:
        dirPath = Path(user_dicts['ld_resume_chkpt']['ld_chkpt']).resolve(
            strict=True).parents[1]
        chkpt_dicts = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        assert len(user_dicts) == len(chkpt_dicts)
        # override  some user_dicts with chkpt_dicts
        for user_dict_k in user_dicts_keys:
            if (not user_dicts[user_dict_k] and
                    user_dict_k != 'ld_resume_chkpt') or\
                    user_dict_k == 'model_init':
                user_dicts[user_dict_k] = chkpt_dicts[user_dict_k]
    elif 'resume_from_checkpoint' in user_dicts['ld_resume_chkpt']:
        if 'resume_from_checkpoint' in user_dicts['trainer']:
            strng = (f'Remove "resume_from_checkpoint" from the "trainer" '
                     f'dictionary in the file {argv[1]}.')
            logg.critical(strng)
            exit()
        dirPath = Path(
            user_dicts['ld_resume_chkpt']['resume_from_checkpoint']).resolve(
                strict=True).parents[1]
        chkpt_dicts = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        assert len(user_dicts) == len(chkpt_dicts)
        # override  some user_dicts with chkpt_dicts
        for user_dict_k in user_dicts_keys:
            if (not user_dicts[user_dict_k] and
                    user_dict_k != 'ld_resume_chkpt') or\
                    user_dict_k == 'model_init' or\
                    user_dict_k == 'optz_sched':
                user_dicts[user_dict_k] = chkpt_dicts[user_dict_k]
        _ = user_dicts['trainer'].pop('resume_from_checkpoint', None)
        user_dicts['trainer']['resume_from_checkpoint'] = user_dicts[
            'ld_resume_chkpt']['resume_from_checkpoint']
    else:
        tb_subDir = ",".join([
            f'{item}={user_dicts["model_init"][item]}'
            for item in ['model_type', 'tokenizer_type']
            if item in user_dicts['model_init']
        ])
        dirPath = Path('tensorboard_logs').joinpath(tb_subDir)
        dirPath.mkdir(parents=True, exist_ok=True)

    # prepare and split dataset
    if user_dicts["model_init"]['model'] == "bert" and user_dicts[
            "model_init"]['tokenizer_type'] == "bert":
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(
            user_dicts['model_init']['model_type'])
    else:
        strng = ('unknown model and tokenizer_type: '
                 f'{user_dicts["model_init"]["model"]}'
                 f'{user_dicts["model_init"]["tokenizer_type"]}')
        logg.critical(strng)
        exit()
    data = Data(tokenizer,
                batch_size=user_dicts['data']['batch_size']
                if 'batch_size' in user_dicts['data'] else {})
    if not ('dataset_path' in user_dicts['data']
            and isinstance(user_dicts['data']['dataset_path'], str)):
        logg.critical('Must specify a path to the dataset.')
        exit()
    data.prepare_data(dataset_path=user_dicts['data']['dataset_path'])
    dataset_metadata = data.split_dataset(
        dataset_path=user_dicts['data']['dataset_path'],
        dataset_split=user_dicts['data']['dataset_split']
        if 'dataset_split' in user_dicts['data'] else {},
        no_training=True if 'no_training' in user_dicts['misc']
        and user_dicts['misc']['no_training'] else False,
        no_testing=True if 'no_testing' in user_dicts['misc']
        and user_dicts['misc']['no_testing'] else False)

    # initialize model
    if 'ld_chkpt' in user_dicts['ld_resume_chkpt']:
        model = Model.load_from_checkpoint(
            checkpoint_path=user_dicts['ld_resume_chkpt']['ld_chkpt'])
        assert model.get_numClasses(
        ) == dataset_metadata['dataset_info']['num_classes']
    else:
        model = Model(user_dicts['model_init'],
                      dataset_metadata['dataset_info']['num_classes'])
    model.params(user_dicts['optz_sched'])

    # create a directory to store all types of results
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
    if not ('no_training' in user_dicts['misc']
            and user_dicts['misc']['no_training']):
        # Training: True, Testing: Don't care
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
        trainer = Trainer(
            logger=tb_logger,
            deterministic=True,
            num_sanity_val_steps=0,
            log_every_n_steps=100,
            accumulate_grad_batches=32,
            #stochastic_weight_avg=True,
            gradient_clip_val=0.5,
            callbacks=[checkpoint_callback, lr_monitor],
            **user_dicts['trainer'])
    elif not ('no_testing' in user_dicts['misc']
              and user_dicts['misc']['no_testing']):
        # Training: False, Testing: True
        trainer = Trainer(logger=True,
                          checkpoint_callback=False,
                          **user_dicts['trainer'])
    else:
        # Training: False, Testing: False
        strng = ('User specified no-training and no-testing. Must do either'
                 'training or testing or both.')
        logg.critical(strng)
        exit()

    # training and testing
    model.kludge(dataset_metadata['batch_size'])
    #trainer.tune(model, datamodule=data)
    if not ('no_training' in user_dicts['misc']
            and user_dicts['misc']['no_training']):
        # Training: True, Testing: Don't care
        trainer.fit(model,
                    train_dataloaders=data.train_dataloader(),
                    val_dataloaders=data.val_dataloader())
        if not ('no_testing' in user_dicts['misc']
                and user_dicts['misc']['no_testing']):
            if 'statistics' in user_dicts['misc'] and user_dicts['misc'][
                    'statistics']:
                model.set_statistics(dataset_metadata, dirPath)
            trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())
            model.clear_statistics()
    elif not ('no_testing' in user_dicts['misc']
              and user_dicts['misc']['no_testing']):
        # Training: False, Testing: True
        if 'statistics' in user_dicts['misc'] and user_dicts['misc'][
                'statistics']:
            model.set_statistics(dataset_metadata, dirPath)
        trainer.test(model, dataloaders=data.test_dataloader())
        model.clear_statistics()
    else:
        # Training: False, Testing: False
        logg.critical('Bug in the Logic')
        exit()
    logg.info(f"Results and other information is at the directory: {dirPath}")


if __name__ == '__main__':
    main()
