'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import pickle
from collections import Counter
import Utilities

logg = getLogger(__name__)


def split_dataset(
    tokenizer, dataset_path: str, splits: Dict[str,
                                               int], batch_sizes: Dict[str,
                                                                       int]
) -> Tuple[Dict[str, Any], List[List[List[Any]]], List[List[List[Any]]],
           List[List[List[Any]]]]:
    assert splits['train'] + splits['val'] + splits['test'] == 100

    # retrieve data files
    dirName = pathlib.Path(dataset_path).resolve(strict=True).parents[0]
    fileName_noSuffix = pathlib.Path(dataset_path).stem
    dataset_file = dirName.joinpath(f'{fileName_noSuffix}.df')
    dataset_meta_file = dirName.joinpath(f'{fileName_noSuffix}.meta')
    if (not dataset_file.exists()) or (not dataset_meta_file.exists()):
        strng = ('Either one or both of following files do not exist: '
                 '{dataset_file}, {dataset_meta_file}')
        logg.critical(strng)
        exit()
    df = pd.read_pickle(dataset_file)
    with dataset_meta_file.open('rb') as dmF:
        bio2_label_names = pickle.load(dmF)

    # Split dataset into train, val, test
    if not splits['train'] and splits['test']:
        # testing a dataset on a checkpoint file; no training
        df_train, df_val, df_test, splits['val'], splits[
            'test'] = None, None, df, 0, 100
    else:
        df_train, df_temp = train_test_split(df,
                                             shuffle=True,
                                             stratify=None,
                                             train_size=(splits['train'] /
                                                         100),
                                             random_state=42)
        df_val, df_test = train_test_split(
            df_temp,
            shuffle=True,
            stratify=None,
            test_size=(splits['test'] / (splits['val'] + splits['test'])),
            random_state=42)
        assert len(df) == len(df_train) + len(df_val) + len(df_test)

    train_data, val_data, test_data = (df_train[[
        'dlg id', 'turn num', 'user input', 'history', 'token labels nums'
    ]].values.tolist() if df_train is not None else None, df_val[[
        'dlg id', 'turn num', 'user input', 'history', 'token labels nums'
    ]].values.tolist() if df_val is not None else None, df_test[[
        'dlg id', 'turn num', 'user input', 'history', 'token labels nums'
    ]].values.tolist() if df_test is not None else None)

    # create meta-data for the datasets
    def token_labels_count(dataset):
        if dataset is None:
            return None
        count = Counter()
        for example in dataset:
            for token_label in example[4]:
                count[token_label] += 1
        return dict(count)

    trainValTest_tokenLabels_count = [
        token_labels_count(dataset)
        for dataset in (train_data, val_data, test_data)
    ]

    # e.g. 'test-set unseen token-labels': token-labels in test-dataset that
    # are not in train-dataset
    def tokens_in_dataset(dataset):
        if dataset is None:
            return set()
        tokens_in_dataset = set()
        for example in dataset:
            tokens_in_dataset |= set(tokenizer(Utilities.preTokenize_splitWords(example[2]), is_split_into_words=True)['input_ids'])
        return tokens_in_dataset

    trainTest_tokens_in_dataset = [
        tokens_in_dataset(dataset)
        for dataset in (train_data, test_data)
    ]
    testSet_unseen_tokens = (trainTest_tokens_in_dataset[1] - trainTest_tokens_in_dataset[0])

    dataset_metadata = {
        'batch sizes': batch_sizes,
        'dataset splits': splits,
        'dataset lengths': {
            'original': len(df),
            'train': len(df_train) if df_train is not None else 0,
            'val': len(df_val) if df_val is not None else 0,
            'test': len(df_test) if df_test is not None else 0
        },
        'token-label names': bio2_label_names,
        'train token-labels -> number:count':
        trainValTest_tokenLabels_count[0],
        'val token-labels -> number:count': trainValTest_tokenLabels_count[1],
        'test token-labels -> number:count': trainValTest_tokenLabels_count[2],
        'test-set unseen tokens': testSet_unseen_tokens,
        'dataset_panda': dataset_file
    }

    return dataset_metadata, train_data, val_data, test_data
