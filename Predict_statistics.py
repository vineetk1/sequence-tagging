'''
Vineet Kumar, sioom.ai
'''

import torch
from logging import getLogger
from typing import Dict, List, Any, Union, Tuple, Set
import pathlib
import textwrap
import pandas as pd
from collections import Counter
from itertools import zip_longest
from Utilities import userOut_init

logg = getLogger(__name__)


def failed_nnOut_tknLblIds(
    bch: Dict[str, Any],
    bch_nnOut_tknLblIds: torch.Tensor,
    bch_userIn_filtered_entityWrds: List[Union[List[str], None]],
    bch_nnOut_entityWrdLbls: List[Union[List[str], None]],
    bch_userOut: List[Dict[str, List[str]]],
    df: pd.DataFrame,
    tokenizer,
    dataset_meta: Dict[str, Any],
    failed_nnOut_tknLblIds: pathlib.Path,
) -> Tuple[int, int, int, int, int, Dict[str, int]]:
    count_total_turns: int = 0
    count_failed_nnOut_tknLblIds: int = 0
    count_failedTurns_nnOut_entityWrdLbl: int = 0
    count_failedTurns_nnOut_userOut: int = 0
    count_failed_turns: int = 0
    counter_failed_nnOut_tknLblIds: Counter(Counter, Dict[str,
                                                          int]) = Counter()

    assert bch_nnOut_tknLblIds.shape[0] == len(bch_nnOut_entityWrdLbls)
    count_total_turns += len(bch_nnOut_entityWrdLbls)
    bch_nnOut_tknLblIds = torch.where(bch['tknLblIds'] == -100,
                                      bch['tknLblIds'], bch_nnOut_tknLblIds)
    failed_bchIdxs_nnOutTknLblIdIdxs: List[List[int, int]] = torch.ne(
        bch['tknLblIds'], bch_nnOut_tknLblIds).nonzero().tolist()
    count_failed_nnOut_tknLblIds += len(failed_bchIdxs_nnOutTknLblIdIdxs)
    failed_bchIdxs: Set[int] = {
        failed_bchIdx_nnOutTknLblIdIdx[0]
        for failed_bchIdx_nnOutTknLblIdIdx in failed_bchIdxs_nnOutTknLblIdIdxs
    }

    bch_entityWrdLbls_True: List[List[str]] = []
    bch_failed_nnOut_entityWrdLbls: List[List[str]] = []
    for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
        bch_entityWrdLbls_True.append([])
        bch_failed_nnOut_entityWrdLbls.append([])
        # ******Finally, must be able to retrieve entityWrdLbls_True from df instead of wrdLbls_True*****
        wrdLbls_True: List[str] = (
            df[(df['dlgId'] == dlgId)
               & (df['trnId'] == trnId)]['wrdLbls']).item()
        for wrdLbl_True in wrdLbls_True:
            if wrdLbl_True[0] == 'B' or wrdLbl_True[0] == 'I':
                if wrdLbl_True[-1] == ')':
                    bch_entityWrdLbls_True[-1].append(
                        wrdLbl_True[2:wrdLbl_True.index('(')])
                else:
                    bch_entityWrdLbls_True[-1].append(wrdLbl_True[2:])
        if bch_nnOut_entityWrdLbls[bch_idx] != bch_entityWrdLbls_True[-1]:
            count_failedTurns_nnOut_entityWrdLbl += 1
            for entityWrdLbl_True, nnOut_entityWrdLbl in zip_longest(
                    bch_entityWrdLbls_True[-1],
                (bch_nnOut_entityWrdLbls[bch_idx]
                 if bch_nnOut_entityWrdLbls[bch_idx] is not None else [])):
                if entityWrdLbl_True != nnOut_entityWrdLbl:
                    bch_failed_nnOut_entityWrdLbls[-1].append(
                        f"({entityWrdLbl_True}, {nnOut_entityWrdLbl})")
    assert len(bch_entityWrdLbls_True) == len(bch_nnOut_entityWrdLbls)

    bch_userOut_True: List[Dict[str, List[str]]] = []
    bch_failed_nnOut_userOut: List[Union[Dict[str, List[str]], None]] = []
    for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
        bch_userOut_True.append(
            (df[(df['dlgId'] == dlgId)
                & (df['trnId'] == trnId)]['userOut']).item())
        if bch_userOut[bch_idx] == bch_userOut_True[-1]:
            bch_failed_nnOut_userOut.append(None)
        else:
            count_failedTurns_nnOut_userOut += 1
            d: dict = userOut_init()
            for k in bch_userOut_True[-1]:
                if bch_userOut[bch_idx][k] != bch_userOut_True[-1][k]:
                    for item_True, item in zip_longest(
                            bch_userOut_True[-1][k], bch_userOut[bch_idx][k]):
                        if item != item_True:
                            d[k].append((item_True, item))
            bch_failed_nnOut_userOut.append(str(d))

    # inner-lists must have same bch_idx occuring consectively
    failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut: List[List[
        int, int]] = []
    for bch_idx in range(len(bch_entityWrdLbls_True)):
        if bch_idx in failed_bchIdxs:
            for failed_bchIdx_nnOutTknLblIdIdx in (
                    failed_bchIdxs_nnOutTknLblIdIdxs):
                if failed_bchIdx_nnOutTknLblIdIdx[0] == bch_idx:
                    failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                        failed_bchIdx_nnOutTknLblIdIdx)
        elif bch_failed_nnOut_entityWrdLbls[bch_idx]:
            failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                [bch_idx, None])
        elif bch_failed_nnOut_userOut[bch_idx] is not None:
            failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                [bch_idx, None])
        else:
            pass

    if not failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut:
        return (count_total_turns, count_failed_nnOut_tknLblIds,
                count_failedTurns_nnOut_entityWrdLbl,
                count_failedTurns_nnOut_userOut, count_failed_turns,
                counter_failed_nnOut_tknLblIds)
    count_failed_turns += len({
        failed_bchIdx_nnOutTknLblIdIdx_entityWrdLbl_userOut[0]
        for failed_bchIdx_nnOutTknLblIdIdx_entityWrdLbl_userOut in
        failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut
    })

    bch_nnIn_tkns: List[List[str]] = []
    bch_unseen_tkns_predictSet: List[List[str]] = []
    bch_tknLbls_True: List[List[str]] = []
    bch_nnOut_tknLbls: List[List[str]] = []
    nnIn_tknIds_beginEnd_idx = (
        bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
    for bch_idx in range(len(bch_nnOut_entityWrdLbls)):
        bch_nnIn_tkns.append([])
        bch_unseen_tkns_predictSet.append([])
        bch_tknLbls_True.append([])
        bch_nnOut_tknLbls.append([])
        index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[bch_idx * 2, 1] + 1
        index_of_second_SEP = nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1, 1]
        for nnIn_tknIds_idx in range(index_of_first_SEP_plus1,
                                     index_of_second_SEP):
            nnIn_tknId: int = (bch['nnIn_tknIds']['input_ids'][bch_idx]
                               [nnIn_tknIds_idx]).item()
            bch_nnIn_tkns[-1].append(
                tokenizer.convert_ids_to_tokens(nnIn_tknId))
            if nnIn_tknId in dataset_meta['test-set unseen tokens']:
                bch_unseen_tkns_predictSet[-1].append(bch_nnIn_tkns[-1][-1])
            bch_tknLbls_True[-1].append(
                dataset_meta['idx2tknLbl'][bch['tknLblIds'][bch_idx,
                                                            nnIn_tknIds_idx]])
            bch_nnOut_tknLbls[-1].append(dataset_meta['idx2tknLbl'][
                bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx]])

    with failed_nnOut_tknLblIds.open('a') as file:
        prev_bch_idx: int = None
        bch_idx: int = None
        wrapper: textwrap.TextWrapper = textwrap.TextWrapper(
            width=80, initial_indent="", subsequent_indent=21 * " ")
        for bch_idx, nnOut_tknLblId_idx in (
                failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut):
            # only FAILED bch_idx and nnOut_tknLblId_idx are considered
            index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[bch_idx * 2,
                                                                1] + 1

            if prev_bch_idx is not None and bch_idx != prev_bch_idx:
                for strng in (
                        f"entityWrdLbls_True = {' '.join(bch_entityWrdLbls_True[prev_bch_idx])}",
                        f"nnOut_entityWrdLbls = {' '.join(bch_nnOut_entityWrdLbls[prev_bch_idx])}"
                        if bch_nnOut_entityWrdLbls[prev_bch_idx] is not None
                        else "nnOut_entityWrdLbls: None",
                        f"Failed-nnOut_entityWrdLbls (entityWrdLbls_True, nnOut_entityWrdLbls): {', '.join(bch_failed_nnOut_entityWrdLbls[prev_bch_idx])}"
                        if bch_failed_nnOut_entityWrdLbls[prev_bch_idx] else
                        "Failed-nnOut_entityWrdLbls: None",
                        f"userIn_filtered_entityWrds = {' '.join(bch_userIn_filtered_entityWrds[prev_bch_idx])}"
                        if bch_userIn_filtered_entityWrds[prev_bch_idx]
                        is not None else "userIn_filtered_entityWrds: None",
                        f"userOut_True = {bch_userOut_True[prev_bch_idx]}",
                        f"nnOut_userOut = {bch_userOut[prev_bch_idx]}",
                        f"Failed-nnOut_userOut (userOut_True, nnOut_userOut): {bch_failed_nnOut_userOut[prev_bch_idx]}"
                        if bch_failed_nnOut_userOut[prev_bch_idx] is not None
                        else "Failed-nnOut_userOut: None",
                        f"Predict-set tkns not seen in train-set = {', '.join(bch_unseen_tkns_predictSet[bch_idx])}"
                        if bch_unseen_tkns_predictSet[bch_idx] else
                        "Predict-set tkns not seen in train-set: None",
                ):
                    file.write(wrapper.fill(strng))
                    file.write("\n")

            if bch_idx != prev_bch_idx:
                # print out: dlgId_trnId, userIn, userIn_filtered wrds,
                # nnIn_tkns, tknLbls_True, and nnOut_tknLbls;
                # tknIds between two SEP belong to tknIds of words in
                # bch['userIn_filtered']
                file.write("\n\n")
                for strng in (
                        f"dlg_id, trn_id = {bch['dlgTrnId'][bch_idx]}",
                        f"userIn = {(df[(df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userIn']).item()}",
                        f"userIn_filtered = {' '.join(bch['userIn_filtered'][bch_idx])}",
                        f"nnIn_tkns = {' '.join(bch_nnIn_tkns[bch_idx])}",
                        f"tknLbls_True = {' '.join(bch_tknLbls_True[bch_idx])}",
                        f"nnOut_tknLbls = {' '.join(bch_nnOut_tknLbls[bch_idx])}",
                        "Failed nnOut_tknLbls (userIn_filtered, nnIn_tkn, tknLbl_True, nnOut_tknLbl):"
                        if nnOut_tknLblId_idx is not None else
                        "Failed nnOut_tknLbls: None",
                ):
                    file.write(wrapper.fill(strng))
                    file.write("\n")

            if nnOut_tknLblId_idx is not None:
                counter_failed_nnOut_tknLblIds[
                    f"{bch_tknLbls_True[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {bch_nnOut_tknLbls[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}"] += 1
                file.write(
                    wrapper.fill(
                        f"{bch['userIn_filtered'][bch_idx][bch['map_tknIdx2wrdIdx'][bch_idx][nnOut_tknLblId_idx]]}, {bch_nnIn_tkns[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {bch_tknLbls_True[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {bch_nnOut_tknLbls[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}  "
                    ))
                file.write("\n")

            prev_bch_idx = bch_idx

        assert count_failed_nnOut_tknLblIds == sum(
            [value for value in counter_failed_nnOut_tknLblIds.values()])
        assert bch_idx is not None
        for strng in (
                f"entityWrdLbls_True = {' '.join(bch_entityWrdLbls_True[bch_idx])}",
                f"nnOut_entityWrdLbls = {' '.join(bch_nnOut_entityWrdLbls[bch_idx])}"
                if bch_nnOut_entityWrdLbls[bch_idx] is not None else
                "nnOut_entityWrdLbls: None",
                f"Failed-nnOut_entityWrdLbls (entityWrdLbls_True, nnOut_entityWrdLbls): {', '.join(bch_failed_nnOut_entityWrdLbls[bch_idx])}"
                if bch_failed_nnOut_entityWrdLbls[bch_idx] else
                "Failed-nnOut_entityWrdLbls: None",
                f"userIn_filtered_entityWrds = {' '.join(bch_userIn_filtered_entityWrds[bch_idx])}"
                if bch_userIn_filtered_entityWrds[bch_idx] is not None else
                "userIn_filtered_entityWrds: None",
                f"userOut_True = {bch_userOut_True[prev_bch_idx]}",
                f"nnOut_userOut = {bch_userOut[prev_bch_idx]}",
                f"Failed-nnOut_userOut (userOut_True, nnOut_userOut): {bch_failed_nnOut_userOut[prev_bch_idx]}"
                if bch_failed_nnOut_userOut[prev_bch_idx] is not None else
                "Failed-nnOut_userOut: None",
                f"Predict-set tkns not seen in train-set = {' '.join(bch_unseen_tkns_predictSet[bch_idx])}"
                if bch_unseen_tkns_predictSet[bch_idx] else
                "Predict-set tkns not seen in train-set: None",
        ):
            file.write(wrapper.fill(strng))
            file.write("\n")

    return (count_total_turns, count_failed_nnOut_tknLblIds,
            count_failedTurns_nnOut_entityWrdLbl,
            count_failedTurns_nnOut_userOut, count_failed_turns,
            counter_failed_nnOut_tknLblIds)


def prepare_metric(
    bch: Dict[str, Any],
    bch_nnOut_tknLblIds: torch.Tensor,
    dataset_meta: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[str]]]:
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    assert bch_nnOut_tknLblIds.shape[0] == bch['tknLblIds'].shape[0]
    # tknIds between two SEP belong to tknIds of words in
    # bch['userIn_filtered']
    nnIn_tknIds_idx_beginEnd: torch.Tensor = (
        bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        y_true.append([])
        y_pred.append([])
        prev_firstTknOfWrd_idx: int = None
        for nnIn_tknIds_idx in range(
            (nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1),
            (nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
            if (firstTknOfWrd_idx :=
                    bch['map_tknIdx2wrdIdx'][bch_idx][nnIn_tknIds_idx]
                ) == prev_firstTknOfWrd_idx:
                continue  # ignore tknId that is not first-token-of-word
            prev_firstTknOfWrd_idx = firstTknOfWrd_idx
            nnOut_tknLbl_True = dataset_meta['idx2tknLbl'][bch['tknLblIds'][
                bch_idx, nnIn_tknIds_idx]]
            #assert nnOut_tknLbl_True != "T"
            assert nnOut_tknLbl_True[0] != "T"
            y_true[-1].append(nnOut_tknLbl_True)
            nnOut_tknLbl = dataset_meta['idx2tknLbl'][bch_nnOut_tknLblIds[
                bch_idx, nnIn_tknIds_idx]]
            #if nnOut_tknLbl == "T":
            if nnOut_tknLbl[0] == "T":
                # "T" is not allowed in the metric, only BIO is allowed;
                # the prediction for first-token-in-word must not be "T";
                # change nnOut_tknLbl so it is not nnOut_tknLbl_True; then
                # nnOut_tknLbl will be considered a wrong prediction
                if nnOut_tknLbl_True[0] == "O":
                    nnOut_tknLbl = f"B{nnOut_tknLbl[1:]}"
                else:
                    nnOut_tknLbl = "O"
            y_pred[-1].append(nnOut_tknLbl)
        assert len(y_true[-1]) == len(y_pred[-1])
    assert len(y_true) == len(y_pred)
    return (y_true, y_pred)


def print_statistics(
    test_results: pathlib.Path,
    dataset_meta: Dict[str, Any],
    count_total_turns: int,
    count_failed_turns: int,
    count_failed_nnOut_tknLblIds: int,
    count_failedTurns_nnOut_entityWrdLbl: int,
    count_failedTurns_nnOut_userOut: int,
    counter_failed_nnOut_tknLblIds: Dict[str, int],
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> None:
    # Print
    from sys import stdout
    from contextlib import redirect_stdout
    from pathlib import Path
    stdoutput = Path('/dev/null')
    for out in (stdoutput, test_results):
        with out.open("a") as results_file:
            with redirect_stdout(stdout if out == stdoutput else results_file):
                turns = (f'# of turns = {count_total_turns}')
                failed_turns = (f'# of failed turns = {count_failed_turns}')
                failed_nnOut_tknLblIds = ('# of failed nnOut_tknLblIds = '
                                          f'{count_failed_nnOut_tknLblIds}')
                failedTurns_nnOut_entityWrdLbl = (
                    '# of turns of failed nnOut_entityWrdLbl = '
                    f'{count_failedTurns_nnOut_entityWrdLbl}')
                failedTurns_nnOut_userOut = (
                    '# of turns of failed nnOut_userOut = '
                    f'{count_failedTurns_nnOut_userOut}')
                for strng in (
                        turns,
                        failed_turns,
                        failed_nnOut_tknLblIds,
                        failedTurns_nnOut_entityWrdLbl,
                        failedTurns_nnOut_userOut,
                ):
                    print(strng)

                strng = ('failed nnOut_tknLblIds '
                         '(tknLbl_True, nnOut_tknLbl: # of times) =')
                print(strng)
                if counter_failed_nnOut_tknLblIds:
                    for k, v in counter_failed_nnOut_tknLblIds.items():
                        print(
                            textwrap.fill(f'{k}: {v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))
                    print('')  # empty line
                else:
                    print('None')

                relevant_keys_of_dataset_meta = [
                    'bch sizes', 'dataset splits', 'dataset lengths',
                    'pandas data-frame file location'
                ]
                for k, v in dataset_meta.items():
                    if k in relevant_keys_of_dataset_meta:
                        print(k)
                        print(
                            textwrap.fill(f'{v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))
                print('')  # empty line

                from seqeval.scheme import IOB2
                from seqeval.metrics import accuracy_score
                from seqeval.metrics import precision_score
                from seqeval.metrics import recall_score
                from seqeval.metrics import f1_score
                from seqeval.metrics import classification_report
                print('Classification Report')
                print(
                    classification_report(y_true,
                                          y_pred,
                                          mode='strict',
                                          scheme=IOB2))
                print('Precision = ', end="")
                print(
                    precision_score(y_true, y_pred, mode='strict',
                                    scheme=IOB2))
                print('Recall = ', end="")
                print(recall_score(y_true, y_pred, mode='strict', scheme=IOB2))
                print('F1 = ', end="")
                print(f1_score(y_true, y_pred, mode='strict', scheme=IOB2))
                strng = ('Accuracy = '
                         f'{accuracy_score(y_true, y_pred): .2f}')
                print(strng)
