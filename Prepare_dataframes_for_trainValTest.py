'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from pathlib import Path
import pickle

logg = getLogger(__name__)


def prepare_dataframes_for_trainValTest(
    tokenizer, dataframes_dirPath: str
) -> Tuple[Dict[str, Any], List[List[List[Any]]], List[List[List[Any]]],
           List[List[List[Any]]]]:

    dataframes_dirPath = Path(dataframes_dirPath).resolve(strict=True)
    df_train_file = dataframes_dirPath.joinpath('train.df')
    df_val_file = dataframes_dirPath.joinpath('val.df')
    df_test_file = dataframes_dirPath.joinpath('test.df')
    df_metadata_file = dataframes_dirPath.joinpath('df_metadata')
    if not (df_train_file.exists() and df_val_file.exists()
            and df_test_file.exists() and df_metadata_file.exists()):
        strng = ('Either one or more of following files do not exist: '
                 '{df_train_file}, {df_val_file}, {df_test_file}, '
                 '{df_metadata_file}')
        logg.critical(strng)
        exit()
    df_train = pd.read_pickle(df_train_file)
    df_val = pd.read_pickle(df_val_file)
    df_test = pd.read_pickle(df_test_file)
    with df_metadata_file.open('rb') as file:
        df_metadata = pickle.load(file)

    train_data, val_data, test_data = (
        df_train[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
                  ]].values.tolist() if df_train is not None else None,
        df_val[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
                ]].values.tolist() if df_val is not None else None,
        df_test[['dlgId', 'trnId', 'userIn', 'prevTrnUserOut', 'tknLblIds'
                 ]].values.tolist() if df_test is not None else None)

    return df_metadata, train_data, val_data, test_data
