'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import random
import copy
import pickle
import gc
import Utilities
from Segments2Sentence import Segments2Sentence
from Other_functions import (remove_type_O_wrdLbls, generate_tknLbls,
                             collect_metric)
from Fill_entityWrds import Fill_entityWrds

logg = getLogger(__name__)


def generate_dataframes(tokenizer, dataframes_dirPath: str,
                        bch_sizes: Dict[str, int]) -> None:
    tknLblId2tknLbl = ["O"]
    dataframes_dirPath = Path(dataframes_dirPath).resolve(strict=True)
    df_train_file = dataframes_dirPath.joinpath('train.df')
    df_val_file = dataframes_dirPath.joinpath('val.df')
    df_test_file = dataframes_dirPath.joinpath('test.df')
    df_metadata_file = dataframes_dirPath.joinpath('df_metadata')

    if (df_train_file.exists() and df_val_file.exists()
            and df_test_file.exists() and df_metadata_file.exists()):
        strng = (
            f'Skipped creating datasets because these files already exist '
            f'at {dataframes_dirPath}')
        logg.info(strng)
        return

    MAX_TURNS_PER_DIALOG: int = 10
    SEGMENTS_PER_SENTENCE: int = None  # default is None => Random
    userIn: str
    userIn_filtered_wrds: List[str]
    wrdLbls: List[str]
    userIn_filtered_entityWrds: List[str]
    entityLbls: List[str]
    fill_entityWrds = Fill_entityWrds(dataframes_dirPath)
    gc.collect()
    dlgId: int = -1
    max_turns_per_dialog: int = MAX_TURNS_PER_DIALOG
    df_train = pd.DataFrame(columns=[
        'dlgId', 'trnId', 'userIn', 'userIn_filtered_entityWrds', 'entityLbls',
        'prevTrnUserOut', 'userOut', 'tknLbls'
    ])
    df_val = pd.DataFrame(columns=[
        'dlgId', 'trnId', 'userIn', 'userIn_filtered_entityWrds', 'entityLbls',
        'prevTrnUserOut', 'userOut', 'tknLbls'
    ])
    df_test = pd.DataFrame(columns=[
        'dlgId', 'trnId', 'userIn', 'userIn_filtered_entityWrds', 'entityLbls',
        'prevTrnUserOut', 'userOut', 'tknLbls'
    ])
    df = {"train": df_train, "val": df_val, "test": df_test}

    for trainValTest in ('train', 'val', 'test'):
        trnId: int = MAX_TURNS_PER_DIALOG
        sentence_from_segs = Segments2Sentence(trainValTest)
        if SEGMENTS_PER_SENTENCE == 1:
            get_segment = sentence_from_segs.get_segment()
        if trainValTest != 'train':
            max_val_or_test_examples = int(len(df_train) * 0.5)
            num_val_or_test_examples = 0

        while ((SEGMENTS_PER_SENTENCE == 1
                and not sentence_from_segs.all_segments_done())
               or (SEGMENTS_PER_SENTENCE != 1 and trainValTest == 'train' and
                   ((not sentence_from_segs.all_segments_done()) or
                    (not fill_entityWrds.all_entityWrds_used())))
               or (SEGMENTS_PER_SENTENCE != 1 and trainValTest != 'train'
                   and num_val_or_test_examples < max_val_or_test_examples)):
            if trnId < max_turns_per_dialog:
                trnId += 1
            else:
                max_turns_per_dialog = random.randint(1, MAX_TURNS_PER_DIALOG)
                trnId = 0
                dlgId = (dlgId + 1) % (10**6)

            if SEGMENTS_PER_SENTENCE != 1:
                sentenceWith_placeholders = sentence_from_segs.get_sentence(
                    segs_per_sentence=SEGMENTS_PER_SENTENCE)
            else:  # SEGMENTS_PER_SENTENCE == 1
                try:
                    sentenceWith_placeholders = next(get_segment)
                except StopIteration:
                    continue

            userIn, userIn_filtered_wrds, wrdLbls = (
                fill_entityWrds.sentence_label(
                    sentenceWith_placeholders=sentenceWith_placeholders,
                    tknLblId2tknLbl=tknLblId2tknLbl,
                ))

            userIn_filtered_entityWrds, entityLbls = remove_type_O_wrdLbls(
                userIn_filtered_wrds=userIn_filtered_wrds, wrdLbls=wrdLbls)

            if trnId == 0:
                prevTrnUserOut: Dict[str, List[str]] = Utilities.userOut_init()

            userOut: Dict[str, List[str]] = Utilities.generate_userOut(
                bch_prevTrnUserOut=[prevTrnUserOut],
                bch_nnOut_userIn_filtered_entityWrds=[
                    userIn_filtered_entityWrds
                ],
                bch_nnOut_entityLbls=[entityLbls])[0]

            history: List[str] = Utilities.prevTrnUserOut2history(
                prevTrnUserOut=prevTrnUserOut)

            tknLbls: List[str] = generate_tknLbls(
                history=history,
                userIn_filtered_wrds=userIn_filtered_wrds,
                wrdLbls=wrdLbls,
                tokenizer=tokenizer)

            row = {
                'dlgId': dlgId,
                'trnId': trnId,
                'userIn': userIn,
                'userIn_filtered_entityWrds': userIn_filtered_entityWrds,
                'entityLbls': entityLbls,
                'prevTrnUserOut': prevTrnUserOut,
                'userOut': userOut,
                'tknLbls': tknLbls
            }
            df[trainValTest].loc[len(df[trainValTest])] = copy.deepcopy(row)
            prevTrnUserOut = df[trainValTest].loc[len(df[trainValTest]) -
                                                  1]['userOut']

            if trainValTest != 'train':
                num_val_or_test_examples += 1

        df[trainValTest]['tknLblIds'] = [[
            tknLblId2tknLbl.index(tknLbl) if tknLbl != -100 else -100
            for tknLbl in tknLbls
        ] for tknLbls in df[trainValTest]['tknLbls']]
        df[trainValTest].drop(columns=['tknLbls'], inplace=True)

    df_metadata: Dict[str, Any] = collect_metric(
        tokenizer,
        SEGMENTS_PER_SENTENCE=SEGMENTS_PER_SENTENCE,
        tknLblId2tknLbl=tknLblId2tknLbl,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test)
    df_metadata['pandas predict-dataframe file location'] = df_test_file
    df_metadata['bch sizes'] = bch_sizes
    df_metadata['entityWrds that have more than one tknLbl'] = (
        fill_entityWrds.get_multilabel_entityWrds())
    df_train.to_pickle(df_train_file)
    df_val.to_pickle(df_val_file)
    df_test.to_pickle(df_test_file)
    with df_metadata_file.open('wb') as file:
        pickle.dump(df_metadata, file, protocol=pickle.HIGHEST_PROTOCOL)
