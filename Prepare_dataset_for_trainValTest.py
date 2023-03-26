'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from pathlib import Path
import pickle
from collections import Counter
import Utilities

logg = getLogger(__name__)


def prepare_dataset_for_trainValTest(
    tokenizer, dataset_dirPath: str, bch_sizes: Dict[str, int]
) -> Tuple[Dict[str, Any], List[List[List[Any]]], List[List[List[Any]]],
           List[List[List[Any]]]]:

    dataset_dirPath = Path(dataset_dirPath).resolve(strict=True)
    df_train_file = dataset_dirPath.joinpath('train.df')
    df_val_file = dataset_dirPath.joinpath('val.df')
    df_test_file = dataset_dirPath.joinpath('test.df')
    tknLblId2tknLbl_file = dataset_dirPath.joinpath('tknLblId2tknLbl')
    if not (df_train_file.exists() and df_val_file.exists()
            and df_test_file.exists() and tknLblId2tknLbl_file.exists()):
        strng = ('Either one or more of following files do not exist: '
                 '{df_train_file}, {df_val_file}, {df_test_file}, '
                 '{tknLblId2tknLbl_file}')
        logg.critical(strng)
        exit()
    df_train = pd.read_pickle(df_train_file)
    df_val = pd.read_pickle(df_val_file)
    df_test = pd.read_pickle(df_test_file)
    with tknLblId2tknLbl_file.open('rb') as file:
        tknLblId2tknLbl = pickle.load(file)

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
        'dataset lengths': {
            'train': len(df_train) if df_train is not None else 0,
            'val': len(df_val) if df_val is not None else 0,
            'test': len(df_test) if df_test is not None else 0
        },
        'tknLblId2tknLbl': tknLblId2tknLbl,
        'train token-labels -> number:count':
        trainValTest_tokenLabels_count[0],
        'val token-labels -> number:count': trainValTest_tokenLabels_count[1],
        'test token-labels -> number:count': trainValTest_tokenLabels_count[2],
        'test-set unseen tokens': testSet_unseen_tkns,
        'pandas predict data-frame file location': df_test_file
    }

    return dataset_metadata, train_data, val_data, test_data
