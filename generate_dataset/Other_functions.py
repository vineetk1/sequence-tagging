'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from collections import Counter
import Utilities
from Synthetic_dataset import (
    unitsLbls,
    cmdsLbls,
    carEntityNonNumLbls,
)

logg = getLogger(__name__)


def remove_type_O_wrdLbls(userIn_filtered_wrds: List[str],
                          wrdLbls: List[str]) -> Tuple[List[str], List[str]]:
    assert len(userIn_filtered_wrds) == len(wrdLbls)
    userIn_filtered_entityWrds: List[str] = []
    entityLbls: List[str] = []
    entityWrd: str = None
    entityLbl: str = None
    multipleWord_entity: str = ""
    for userIn_filtered_wrd, wrdLbl in zip(userIn_filtered_wrds, wrdLbls):
        assert wrdLbl[0] == 'O' or wrdLbl[0] == 'B' or wrdLbl[0] == 'I'
        if wrdLbl[0] == 'O':
            if multipleWord_entity:  # previous multipleWord_entity
                userIn_filtered_entityWrds.append(multipleWord_entity)
            multipleWord_entity = ""
            continue
        if wrdLbl[-1] == ')':
            # there MUST be an opening-parenthesis, else error
            wrdLbl_openParen_idx = wrdLbl.index('(')
            entityWrd = wrdLbl[wrdLbl_openParen_idx + 1:-1]
            entityLbl = wrdLbl[2:wrdLbl_openParen_idx]
        else:
            entityWrd = userIn_filtered_wrd
            entityLbl = wrdLbl[2:]
        if wrdLbl[0] == 'B':
            if multipleWord_entity:  # previous multipleWord_entity
                userIn_filtered_entityWrds.append(multipleWord_entity)
            entityLbls.append(entityLbl)
            multipleWord_entity = entityWrd
        else:  # wrdLbl[0] is 'I'
            assert multipleWord_entity
            assert entityLbls[-1] == entityLbl
            multipleWord_entity = f"{multipleWord_entity} {entityWrd}"
    if multipleWord_entity:  # previous multipleWord_entity
        userIn_filtered_entityWrds.append(multipleWord_entity)
    assert len(userIn_filtered_entityWrds) == len(entityLbls)
    return userIn_filtered_entityWrds, entityLbls


def generate_tknLbls(history: List[str], userIn_filtered_wrds: List[str],
                     wrdLbls: List[str], tokenizer) -> List[str]:
    tknLbls: List[str] = []
    assert len(userIn_filtered_wrds) == len(wrdLbls)
    assert wrdLbls

    tknIds = tokenizer(
        text=history,
        text_pair=userIn_filtered_wrds,
        is_split_into_words=True,
        padding=True,  # it won't happen anyway
        truncation='do_not_truncate',
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_overflowing_tokens=False)
    map_tknIdx2wrdIdx = tknIds.word_ids(0)
    assert map_tknIdx2wrdIdx.count(None) == 3
    tknIds['input_ids'].shape[1] == len(map_tknIdx2wrdIdx)

    tknLbls = [-100]  # takes care of map_tknIdx2wrdIdx[0], which has None
    for history_tknId_idx in range(1, len(map_tknIdx2wrdIdx)):
        if map_tknIdx2wrdIdx[history_tknId_idx] is not None:
            tknLbls.append(-100)
        else:
            break
    for userInFilteredWrds_tknId_idx in range(history_tknId_idx,
                                              len(map_tknIdx2wrdIdx)):
        # tknId = tknIds['input_ids'][0, userInFilteredWrds_tknId_idx].item()
        # userInFilteredWrds_idx =
        #                       map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx]
        # userIn_filtered_wrd = userIn_filtered_wrds[userInFilteredWrds_idx]
        # wrdLbl = wrdLbls[userInFilteredWrds_idx]
        # Assume no indexing problem in the following two cases:
        # (1) map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx] !=
        # map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx-1] => first
        #                                                       token of a word
        # (2) map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx] !=
        #            map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx+1] => last
        #                                                       token of a word
        userInFilteredWrds_idx = map_tknIdx2wrdIdx[
            userInFilteredWrds_tknId_idx]
        wrdLbl = (wrdLbls[userInFilteredWrds_idx]
                  if userInFilteredWrds_idx is not None else None)
        prev_userInFilteredWrds_idx = map_tknIdx2wrdIdx[
            userInFilteredWrds_tknId_idx - 1]
        if userInFilteredWrds_idx is None:
            # special-token (e.g. CLS, SEP, PAD) gets a label of -100
            tknLbls.append(-100)
        elif userInFilteredWrds_idx != prev_userInFilteredWrds_idx:
            # first token of a word gets the label of that word;
            # no indexing error because
            #         map_tknIdx2wrdIdx[userInFilteredWrds_tknId_idx-1=0]
            #         is always None and this case is handled by "if statement"
            tknLbls.append(wrdLbl)
        else:  # userInFilteredWrds_idx == prev_userInFilteredWrds_idx
            # if not first token then a remaining token of that word
            tknLbls.append(-100)
    assert tknIds['input_ids'][0, 0] == 101  # CLS at index 0
    assert tknLbls[0] == -100
    indices_of_two_SEP = (tknIds['input_ids'] == 102).nonzero()
    assert indices_of_two_SEP.shape[0] == 2
    assert tknLbls[indices_of_two_SEP[0, 1]] == -100
    assert tknLbls[indices_of_two_SEP[1, 1]] == -100
    assert len(tknLbls) == tknIds['input_ids'].shape[1]
    assert len(tknLbls) == tknIds['attention_mask'].count_nonzero(-1).item()
    return tknLbls


def collect_metric(tokenizer, SEGMENTS_PER_SENTENCE: int,
                   tknLblId2tknLbl: List[str], df_train: pd.DataFrame,
                   df_val: pd.DataFrame,
                   df_test: pd.DataFrame) -> Dict[str, Any]:

    # create meta-data for the dataframes
    # generate count of each tknLblId; i.e. tknLblId0: count0, etc.
    def tknLblIds_count(col_tknLblIds):
        count = Counter()
        for row_tknLblIds in col_tknLblIds:
            for tknLblId in row_tknLblIds:
                if tknLblId != -100:
                    count[tknLblId] += 1
        return dict(count)

    trainValTest_tknLblId_count = [
        tknLblIds_count(col_tknLblIds)
        for col_tknLblIds in (df_train['tknLblIds'], df_val['tknLblIds'],
                              df_test['tknLblIds'])
    ]
    assert len(trainValTest_tknLblId_count[0]) == len(
        tknLblId2tknLbl) if SEGMENTS_PER_SENTENCE != 1 else True
    assert 0 not in trainValTest_tknLblId_count[0].values()
    assert -100 not in tknLblId2tknLbl

    # e.g. 'test-set unseen token-labels': token-labels in test-dataframe that
    # are not in train-dataframe
    def entityWrds_in_dataframe(col_entityWrds, col_entityLbls):
        wrds_in_df = set()
        for row_wrds, row_lbls in zip(col_entityWrds, col_entityLbls):
            assert len(row_wrds) == len(row_lbls)
            wrds_in_df |= {
                chunk_wrds
                for chunk_wrds, lbl in zip(row_wrds, row_lbls)
                if lbl in (tuple(unitsLbls.keys()) + tuple(cmdsLbls.keys()) +
                           carEntityNonNumLbls)
            }
        return wrds_in_df

    trainTest_tkns_in_dataframe = [
        entityWrds_in_dataframe(df['userIn_filtered_entityWrds'],
                                df['entityLbls']) for df in (df_train, df_test)
    ]
    testSet_unseen_entityWrds = tuple(
        (trainTest_tkns_in_dataframe[1] - trainTest_tkns_in_dataframe[0]))

    return {
        '# of dialog-turns in dataframes': {
            'train': df_train.shape[0],
            'val': df_val.shape[0],
            'test': df_test.shape[0],
        },
        'tknLblId2tknLbl': tuple(tknLblId2tknLbl),
        'train-set tknLblIds:count': trainValTest_tknLblId_count[0],
        'val-set tknLblIds:count': trainValTest_tknLblId_count[1],
        'test-set tknLblIds:count': trainValTest_tknLblId_count[2],
        'predict-set entityWrds not seen in train-set':
        testSet_unseen_entityWrds,
        'pandas predict-dataframe file location': None
    }
