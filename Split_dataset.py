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
    tokenizer, dataset_dirPath: str, splits: Dict[str,
                                                  int], bch_sizes: Dict[str,
                                                                        int]
) -> Tuple[Dict[str, Any], List[List[List[Any]]], List[List[List[Any]]],
           List[List[List[Any]]]]:
    assert splits['train'] + splits['val'] + splits['test'] == 100

    # retrieve data files
    dataset_dirPath = pathlib.Path(dataset_dirPath).resolve(strict=True)
    dataset_file = dataset_dirPath.joinpath('dataset.df')
    dataset_meta_file = dataset_dirPath.joinpath('dataset.meta')
    if (not dataset_file.exists()) or (not dataset_meta_file.exists()):
        strng = ('Either one or both of following files do not exist: '
                 '{dataset_file}, {dataset_meta_file}')
        logg.critical(strng)
        exit()
    df = pd.read_pickle(dataset_file)
    with dataset_meta_file.open('rb') as dmF:
        idx2tknLbl = pickle.load(dmF)

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

    train_data, val_data, test_data = (
            df_train[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
    ]].values.tolist() if df_train is not None else None,
            df_val[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
    ]].values.tolist() if df_val is not None else None,
            df_test[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
    ]].values.tolist() if df_test is not None else None)

    # create meta-data for the datasets
    def tknLbls_count(dataset):
        if dataset is None:
            return None
        count = Counter()
        for example in dataset:
            for tknLbl in example[4]:
                count[tknLbl] += 1
        return dict(count)

    trainValTest_tokenLabels_count = [
        tknLbls_count(dataset) for dataset in (train_data, val_data, test_data)
    ]

    # e.g. 'test-set unseen token-labels': token-labels in test-dataset that
    # are not in train-dataset
    def tkns_in_dataset(dataset):
        if dataset is None:
            return set()
        tkns_in_dataset = set()
        for example in dataset:
            tkns_in_dataset |= set(
                tokenizer(Utilities.userIn_filter_splitWords(example[2]),
                          is_split_into_words=True)['input_ids'])
        return tkns_in_dataset

    trainTest_tkns_in_dataset = [
        tkns_in_dataset(dataset) for dataset in (train_data, test_data)
    ]
    testSet_unseen_tkns = (trainTest_tkns_in_dataset[1] -
                           trainTest_tkns_in_dataset[0])

    dataset_metadata = {
        'bch sizes': bch_sizes,
        'dataset splits': splits,
        'dataset lengths': {
            'original': len(df),
            'train': len(df_train) if df_train is not None else 0,
            'val': len(df_val) if df_val is not None else 0,
            'test': len(df_test) if df_test is not None else 0
        },
        'idx2tknLbl': idx2tknLbl,
        'train token-labels -> number:count':
        trainValTest_tokenLabels_count[0],
        'val token-labels -> number:count': trainValTest_tokenLabels_count[1],
        'test token-labels -> number:count': trainValTest_tokenLabels_count[2],
        'test-set unseen tokens': testSet_unseen_tkns,
        'dataset_panda': dataset_file
    }

    return dataset_metadata, train_data, val_data, test_data
