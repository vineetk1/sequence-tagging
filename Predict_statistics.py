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
    bch_nnOut_userIn_filtered_entityWrds: List[Union[List[str], None]],
    bch_nnOut_entityLbls: List[Union[List[str], None]],
    bch_nnOut_userOut: List[Dict[str, List[str]]],
    df: pd.DataFrame,
    tokenizer,
    dataset_meta: Dict[str, Any],
    failed_nnOut_tknLblIds_file: pathlib.Path,
    failed_nnOut_entityLblsUserOut_file: pathlib.Path,
    passed_file: pathlib.Path,
) -> Tuple[int, int, int, int, int, Dict[str, int]]:
    count_total_turns: int = 0
    count_failed_nnOut_tknLblIds: int = 0
    count_failedTurns_nnOut_tknLblIds: int = 0
    count_failedTurns_nnOut_entityLbl: int = 0
    count_failedTurns_nnOut_userOut: int = 0
    count_failed_turns: int = 0
    counter_failed_nnOut_tknLblIds: Counter(Counter, Dict[str,
                                                          int]) = Counter()

    assert bch_nnOut_tknLblIds.shape[0] == len(bch_nnOut_entityLbls)
    count_total_turns += len(bch_nnOut_entityLbls)
    bch_nnOut_tknLblIds = torch.where(bch['tknLblIds'] == -100,
                                      bch['tknLblIds'], bch_nnOut_tknLblIds)
    failed_bchIdxs_nnOutTknLblIdIdxs: List[List[int, int]] = torch.ne(
        bch['tknLblIds'], bch_nnOut_tknLblIds).nonzero().tolist()
    count_failed_nnOut_tknLblIds += len(failed_bchIdxs_nnOutTknLblIdIdxs)
    failed_bchIdxs: Set[int] = {
        failed_bchIdx_nnOutTknLblIdIdx[0]
        for failed_bchIdx_nnOutTknLblIdIdx in failed_bchIdxs_nnOutTknLblIdIdxs
    }
    count_failedTurns_nnOut_tknLblIds += len(failed_bchIdxs)

    bch_userIn_filtered_entityWrds_True: List[List[str]] = []
    bch_idx: int
    for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
        bch_userIn_filtered_entityWrds_True.append(
            df[(df['dlgId'] == dlgId)
               & (df['trnId'] == trnId)]['userIn_filtered_entityWrds'].item())

    bch_entityLbls_True: List[List[str]] = []
    bch_failed_nnOut_entityLbls: List[List[str]] = []
    for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
        bch_failed_nnOut_entityLbls.append([])
        bch_entityLbls_True.append(
            df[(df['dlgId'] == dlgId)
               & (df['trnId'] == trnId)]['entityLbls'].item())
        if bch_nnOut_entityLbls[bch_idx] != bch_entityLbls_True[-1]:
            count_failedTurns_nnOut_entityLbl += 1
            for entityLbl_True, nnOut_entityLbl in zip_longest(
                    bch_entityLbls_True[-1],
                (bch_nnOut_entityLbls[bch_idx]
                 if bch_nnOut_entityLbls[bch_idx] is not None else [])):
                if entityLbl_True != nnOut_entityLbl:
                    bch_failed_nnOut_entityLbls[-1].append(
                        f"({entityLbl_True}, {nnOut_entityLbl})")
    assert len(bch_entityLbls_True) == len(bch_nnOut_entityLbls)

    bch_userOut_True: List[Dict[str, List[str]]] = []
    bch_failed_nnOut_userOut: List[Union[Dict[str, List[str]], None]] = []
    for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
        bch_userOut_True.append(
            (df[(df['dlgId'] == dlgId)
                & (df['trnId'] == trnId)]['userOut']).item())
        if bch_nnOut_userOut[bch_idx] == bch_userOut_True[-1]:
            bch_failed_nnOut_userOut.append(None)
        else:
            count_failedTurns_nnOut_userOut += 1
            d: dict = userOut_init()
            for k in bch_userOut_True[-1]:
                if bch_nnOut_userOut[bch_idx][k] != bch_userOut_True[-1][k]:
                    for item_True, item in zip_longest(
                            bch_userOut_True[-1][k],
                            bch_nnOut_userOut[bch_idx][k]):
                        if item != item_True:
                            d[k].append((item_True, item))
            bch_failed_nnOut_userOut.append(str(d))

    bch_nnIn_tkns: List[List[str]] = []
    bch_unseen_tkns_predictSet: List[List[str]] = []
    bch_tknLbls_True: List[List[str]] = []
    bch_nnOut_tknLbls: List[List[str]] = []
    nnIn_tknIds_beginEnd_idx = (
        bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
    for bch_idx in range(len(bch_nnOut_entityLbls)):
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
            bch_tknLbls_True[-1].append(dataset_meta['tknLblId2tknLbl'][
                bch['tknLblIds'][bch_idx, nnIn_tknIds_idx]])
            bch_nnOut_tknLbls[-1].append(dataset_meta['tknLblId2tknLbl'][
                bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx]])

    with failed_nnOut_tknLblIds_file.open(
            'a') as file_failed, failed_nnOut_entityLblsUserOut_file.open(
                'a') as file_halfFailed, passed_file.open('a') as file_passed:

        # append[bch_idx, print_preamble, failed_nnOutTknLblIdIdx,
        # print_nnOutEntityLblUserOut, file_name]
        # nnOutTknLblIdIdxs failed for some bch_idxs
        failed_items: List[List[int, bool, int, bool, str]] = []
        for idx in range(len(failed_bchIdxs_nnOutTknLblIdIdxs)):
            if idx == 0:
                failed_items.append([
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][0], True,
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][1], False,
                    file_failed
                ])
            elif failed_bchIdxs_nnOutTknLblIdIdxs[idx][
                    0] != failed_bchIdxs_nnOutTknLblIdIdxs[idx - 1][0]:
                failed_items[-1][3] = True
                failed_items.append([
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][0], True,
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][1], False,
                    file_failed
                ])
            else:
                failed_items.append([
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][0], False,
                    failed_bchIdxs_nnOutTknLblIdIdxs[idx][1], False,
                    file_failed
                ])
        if failed_items:
            failed_items[-1][3] = True
        count_failed_turns += len({
            failed_bchIdx_nnOutTknLblIdIdx[0]
            for failed_bchIdx_nnOutTknLblIdIdx in
            failed_bchIdxs_nnOutTknLblIdIdxs
        })
        # nnOutTknLblIdIdxs passed for some bch_idxs but
        # nnOut_entityLbls or nnOut_userOut could have failed
        for bch_idx in range(len(bch_entityLbls_True)):
            if bch_idx not in failed_bchIdxs:
                if bch_failed_nnOut_entityLbls[bch_idx] or (
                        bch_failed_nnOut_userOut[bch_idx] is not None):
                    failed_items.append(
                        [bch_idx, True, None, True, file_halfFailed])
                    count_failed_turns += 1
                else:
                    failed_items.append(
                        [bch_idx, True, None, True, file_passed])

        print_to_file: Dict[str, pathlib.Path] = {"file": None}
        wrapper: textwrap.TextWrapper = textwrap.TextWrapper(
            width=80, initial_indent="", subsequent_indent=21 * " ")

        for bch_idx, print_preamble, failed_nnOutTknLblIdIdx,\
                print_nnOutEntityLblUserOut,\
                print_to_file["file"] in failed_items:
            index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[bch_idx * 2,
                                                                1] + 1
            if print_preamble:
                # print out: dlgId_trnId, userIn, userIn_filtered wrds,
                # nnIn_tkns, tknLbls_True, and nnOut_tknLbls;
                # tknIds between two SEP belong to tknIds of words in
                # bch['userIn_filtered_wrds']
                print_to_file["file"].write("\n\n")
                for strng in (
                        f"dlg_id, trn_id = {bch['dlgTrnId'][bch_idx]}",
                        f"userIn = {(df[(df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userIn']).item()}",
                        f"userIn_filtered_wrds = {' '.join(bch['userIn_filtered_wrds'][bch_idx])}",
                        f"nnIn_tkns = {' '.join(bch_nnIn_tkns[bch_idx])}",
                        f"tknLbls_True = {' '.join(bch_tknLbls_True[bch_idx])}",
                        f"nnOut_tknLbls = {' '.join(bch_nnOut_tknLbls[bch_idx])}",
                        "Failed nnOut_tknLbls (userIn_filtered.., nnIn_tkn, tknLbl_True, nnOut_tknLbl):"
                        if failed_nnOutTknLblIdIdx is not None else
                        "Failed nnOut_tknLbls: None",
                ):
                    print_to_file["file"].write(wrapper.fill(strng))
                    print_to_file["file"].write("\n")

            if failed_nnOutTknLblIdIdx is not None:
                counter_failed_nnOut_tknLblIds[
                    f"{bch_tknLbls_True[bch_idx][failed_nnOutTknLblIdIdx - index_of_first_SEP_plus1]}, {bch_nnOut_tknLbls[bch_idx][failed_nnOutTknLblIdIdx - index_of_first_SEP_plus1]}"] += 1
                print_to_file["file"].write(
                    wrapper.fill(
                        f"{bch['userIn_filtered_wrds'][bch_idx][bch['map_tknIdx2wrdIdx'][bch_idx][failed_nnOutTknLblIdIdx]]}, {bch_nnIn_tkns[bch_idx][failed_nnOutTknLblIdIdx - index_of_first_SEP_plus1]}, {bch_tknLbls_True[bch_idx][failed_nnOutTknLblIdIdx - index_of_first_SEP_plus1]}, {bch_nnOut_tknLbls[bch_idx][failed_nnOutTknLblIdIdx - index_of_first_SEP_plus1]}  "
                    ))
                print_to_file["file"].write("\n")

            if print_nnOutEntityLblUserOut:
                for strng in (
                        f"userIn_filtered_entityWrds_True = {' '.join(bch_userIn_filtered_entityWrds_True[bch_idx])}",
                        f"nnOut_userIn_filtered_entityWrds = {' '.join(bch_nnOut_userIn_filtered_entityWrds[bch_idx])}"
                        if bch_nnOut_userIn_filtered_entityWrds[bch_idx] else
                        "nnOut_userIn_filtered_entityWrds: None",
                        f"entityLbls_True = {' '.join(bch_entityLbls_True[bch_idx])}",
                        f"nnOut_entityLbls = {' '.join(bch_nnOut_entityLbls[bch_idx])}"
                        if bch_nnOut_entityLbls[bch_idx] else
                        "nnOut_entityLbls: None",
                        f"Failed-nnOut_entityLbls (entityLbls_True, nnOut_entityLbls): {', '.join(bch_failed_nnOut_entityLbls[bch_idx])}"
                        if bch_failed_nnOut_entityLbls[bch_idx] else
                        "Failed-nnOut_entityLbls: None",
                        f"userOut_True = {bch_userOut_True[bch_idx]}",
                        f"nnOut_userOut = {bch_nnOut_userOut[bch_idx]}",
                        f"Failed-nnOut_userOut (userOut_True, nnOut_userOut): {bch_failed_nnOut_userOut[bch_idx]}"
                        if bch_failed_nnOut_userOut[bch_idx] is not None else
                        "Failed-nnOut_userOut: None",
                        f"Predict-set tkns not seen in train-set = {', '.join(bch_unseen_tkns_predictSet[bch_idx])}"
                        if bch_unseen_tkns_predictSet[bch_idx] else
                        "Predict-set tkns not seen in train-set: None",
                ):
                    print_to_file["file"].write(wrapper.fill(strng))
                    print_to_file["file"].write("\n")

        assert count_failed_nnOut_tknLblIds == sum(
            [value for value in counter_failed_nnOut_tknLblIds.values()])
        assert bch_idx is not None

    return (count_total_turns, count_failed_nnOut_tknLblIds,
            count_failedTurns_nnOut_tknLblIds,
            count_failedTurns_nnOut_entityLbl, count_failedTurns_nnOut_userOut,
            count_failed_turns, counter_failed_nnOut_tknLblIds)


def prepare_metric(
    bch: Dict[str, Any],
    bch_nnOut_tknLblIds: torch.Tensor,
    dataset_meta: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[str]]]:
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    assert bch_nnOut_tknLblIds.shape[0] == bch['tknLblIds'].shape[0]
    # tknIds between two SEP belong to tknIds of words in
    # bch['userIn_filtered_wrds']
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
            nnOut_tknLbl_True = dataset_meta['tknLblId2tknLbl'][
                bch['tknLblIds'][bch_idx, nnIn_tknIds_idx]]
            #assert nnOut_tknLbl_True != "T"
            assert nnOut_tknLbl_True[0] != "T"
            y_true[-1].append(nnOut_tknLbl_True)
            nnOut_tknLbl = dataset_meta['tknLblId2tknLbl'][bch_nnOut_tknLblIds[
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
    count_failedTurns_nnOut_tknLblIds: int,
    count_failedTurns_nnOut_entityLbl: int,
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
                failedTurns_nnOut_tknLblIds = (
                    '# of turns of failed nnOut_tknLblIds = '
                    f'{count_failedTurns_nnOut_tknLblIds}')
                failedTurns_nnOut_entityLbl = (
                    '# of turns of failed nnOut_entityLbl = '
                    f'{count_failedTurns_nnOut_entityLbl}')
                failedTurns_nnOut_userOut = (
                    '# of turns of failed nnOut_userOut = '
                    f'{count_failedTurns_nnOut_userOut}')
                for strng in (
                        turns,
                        failed_turns,
                        failed_nnOut_tknLblIds,
                        failedTurns_nnOut_tknLblIds,
                        failedTurns_nnOut_entityLbl,
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
                    'bch sizes', 'dataset lengths',
                    'pandas predict data-frame file location'
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
