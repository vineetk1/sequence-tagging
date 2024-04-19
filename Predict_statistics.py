'''
Vineet Kumar, sioom.ai
'''

import torch
from logging import getLogger
from typing import Dict, List, Any, Union, Tuple
import pathlib
import textwrap
import pandas as pd
from itertools import zip_longest
from Utilities import userOut_init
from contextlib import redirect_stdout

logg = getLogger(__name__)


def failed_nnOut_tknLblIds(
    bch: Dict[str, Any],
    bch_nnOut_tknLblIds: torch.Tensor,
    bch_nnOut_userIn_filtered_entityWrds: List[Union[List[str], None]],
    bch_nnOut_entityLbls: List[Union[List[str], None]],
    bch_nnOut_userOut: List[Dict[str, List[str]]],
    df: pd.DataFrame,
    tokenizer,
    dataframes_meta: Dict[str, Any],
    count: Dict[str, Union[int, Dict[str, int]]],
    failed_nnOut_tknLblIds_file: pathlib.Path,
    passed_file: pathlib.Path,
) -> Tuple[int, int, int, int, int, Dict[str, int]]:

    assert bch_nnOut_tknLblIds.shape[0] == len(bch_nnOut_entityLbls)
    count["total_turns"] += len(bch_nnOut_entityLbls)

    for bch_idx in range(len(bch_nnOut_entityLbls)):

        # associate (nnIn_tkn, tknLbl_True, nnOut_tknLbl) and
        # userIn_filtered_wrds_True
        tknLbls_assoc: List[str] = []
        prev_wrd_idx: int = None
        next_wrd_idx: int = None
        tkns: str = None
        tknLbl_True: str = None
        nnOut_tknLbl: str = None
        tknLbl_Pass: bool = True
        userIn_filtered_wrds_True: List[str] = []
        nnIn_tknIds_beginEnd: torch.Tensor = (
            bch['nnIn_tknIds']['input_ids'][bch_idx] == 102).nonzero()
        for nnIn_tknIds_idx in range(nnIn_tknIds_beginEnd[0].item() + 1,
                                     nnIn_tknIds_beginEnd[1].item()):
            nnIn_tkn: str = tokenizer.convert_ids_to_tokens(
                bch['nnIn_tknIds']['input_ids'][bch_idx]
                [nnIn_tknIds_idx].item())
            if (next_wrd_idx := bch['map_tknIdx2wrdIdx'][bch_idx]
                [nnIn_tknIds_idx]) != prev_wrd_idx:
                # ignore tknId that is not first-token-of-the-word; BIO labels
                assert not nnIn_tkn.startswith("##")

                # create (nnIn_tkn, tknLbl_True, nnOut_tknLbl)
                if tkns is not None:
                    if tknLbl_True != nnOut_tknLbl:
                        if tknLbl_True[-1] != ")" and nnOut_tknLbl[-1] == ")":
                            nnOut_tknLbl_openParen_idx = nnOut_tknLbl.index(
                                '(')
                            if ((tknLbl_True
                                 == nnOut_tknLbl[:nnOut_tknLbl_openParen_idx])
                                    and
                                (tkns.translate({ord(i): None
                                                 for i in '# '})
                                 == (nnOut_tknLbl[nnOut_tknLbl_openParen_idx +
                                                  1:-1]))):
                                nnOut_tknLbl_pass = True
                            else:
                                nnOut_tknLbl_pass = False
                        else:
                            nnOut_tknLbl_pass = False
                    else:
                        nnOut_tknLbl_pass = True
                    if nnOut_tknLbl_pass:
                        tknLbls_assoc.append(
                            f'({tkns}, {tknLbl_True}, {nnOut_tknLbl})')
                    else:
                        tknLbls_assoc.append(
                            f'({"++t+ "} {tkns}, {tknLbl_True}, '
                            f'{nnOut_tknLbl})')
                        tknLbl_Pass = False
                        count["failed_tknLbls"] += 1
                tkns = f'{nnIn_tkn}'
                tknLblId_True: int = bch['tknLblIds'][bch_idx,
                                                      nnIn_tknIds_idx].item()
                assert tknLblId_True != -100
                tknLbl_True = dataframes_meta['tknLblId2tknLbl'][tknLblId_True]
                nnOut_tknLblId: int = bch_nnOut_tknLblIds[
                    bch_idx, nnIn_tknIds_idx].item()
                assert nnOut_tknLblId != -100
                nnOut_tknLbl = dataframes_meta['tknLblId2tknLbl'][
                    nnOut_tknLblId]
                if tknLblId_True in count["failed_tknLbls_perDlg"]:
                    if nnOut_tknLblId in count["failed_tknLbls_perDlg"][
                            tknLblId_True]:
                        count["failed_tknLbls_perDlg"][tknLblId_True][
                            nnOut_tknLblId] += 1
                    else:
                        count["failed_tknLbls_perDlg"][tknLblId_True][
                            nnOut_tknLblId] = 1
                else:
                    count["failed_tknLbls_perDlg"][tknLblId_True] = {
                        nnOut_tknLblId: 1
                    }

                # create userIn_filtdred_wrds_True
                if tknLbl_True[-1] != ")":
                    userIn_filtered_wrds_True.append(
                        bch['userIn_filtered_wrds'][bch_idx]
                        [bch['map_tknIdx2wrdIdx'][bch_idx][nnIn_tknIds_idx]])
                else:
                    tknLbl_True = dataframes_meta['tknLblId2tknLbl'][
                        tknLblId_True]
                    userIn_filtered_wrds_True.append(
                        tknLbl_True[tknLbl_True.index('(') + 1:-1])

            else:
                assert tkns is not None
                tkns = f'{tkns} {nnIn_tkn}'
            prev_wrd_idx = next_wrd_idx
        if tkns is not None:
            if tknLbl_True != nnOut_tknLbl:
                if tknLbl_True[-1] != ")" and nnOut_tknLbl[-1] == ")":
                    nnOut_tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                    if ((tknLbl_True
                         == nnOut_tknLbl[:nnOut_tknLbl_openParen_idx])
                            and (tkns.translate({ord(i): None
                                                 for i in '# '})
                                 == (nnOut_tknLbl[nnOut_tknLbl_openParen_idx +
                                                  1:-1]))):
                        nnOut_tknLbl_pass = True
                    else:
                        nnOut_tknLbl_pass = False
                else:
                    nnOut_tknLbl_pass = False
            else:
                nnOut_tknLbl_pass = True
            if nnOut_tknLbl_pass:
                tknLbls_assoc.append(
                    f'({tkns}, {tknLbl_True}, {nnOut_tknLbl})')
            else:
                tknLbls_assoc.append(
                    f'({"++t+ "} {tkns}, {tknLbl_True}, {nnOut_tknLbl})')
                tknLbl_Pass = False
                count["failed_tknLbls"] += 1
        if not tknLbl_Pass:
            count["failedTurns_tknLbls"] += 1

        # associate (userIn_filtered_entityWrds_True, bch_entityLbls_True,
        #            nnOut_userIn_filtered_entityWrds, bch_nnOut_entityLbls)
        entityLbl_assoc: List[str] = []
        entityLbl_Pass: bool = None
        userIn_filtered_entityWrds_True: List[str] = df[
            (df['dlgId'] == bch['dlgTrnId'][bch_idx][0])
            & (df['trnId'] == bch['dlgTrnId'][bch_idx][1]
               )]['userIn_filtered_entityWrds'].item()
        entityLbls_True: List[str] = df[
            (df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) &
            (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['entityLbls'].item()
        if bch_nnOut_entityLbls[bch_idx] != entityLbls_True:
            count["failedTurns_entityLbls"] += 1
            entityLbl_Pass = False
        else:
            entityLbl_Pass = True
        for entityWrd_True, entityLbl_True, entityWrd, entityLbl in (
                zip_longest(
                    userIn_filtered_entityWrds_True, entityLbls_True,
                    (bch_nnOut_userIn_filtered_entityWrds[bch_idx]
                     if bch_nnOut_userIn_filtered_entityWrds[bch_idx]
                     is not None else []),
                    (bch_nnOut_entityLbls[bch_idx]
                     if bch_nnOut_entityLbls[bch_idx] is not None else []))):
            entityLbl_assoc.append(
                f'({"++e+ " if entityLbl_True != entityLbl else ""}'
                f'{entityWrd_True}, {entityLbl_True}, {entityWrd}, '
                f'{entityLbl})')

        # userOut_True vs. nnOut_userOut
        failed_nnOut_userOut: Union[Dict[str, List[str], None]]
        userOut_Pass: bool = None
        userOut_True: Dict[str, List[str]] = df[
            (df['dlgId'] == bch['dlgTrnId'][bch_idx][0])
            & (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userOut'].item()
        if bch_nnOut_userOut[bch_idx] == userOut_True:
            failed_nnOut_userOut: Union[Dict[str, List[str], None]] = None
            userOut_Pass = True
        else:
            userOut_Pass = False
            count["failedTurns_userOut"] += 1
            d: dict = userOut_init()
            for k in userOut_True:
                if bch_nnOut_userOut[bch_idx][k] != userOut_True[k]:
                    for item_True, item in zip_longest(
                            userOut_True[k], bch_nnOut_userOut[bch_idx][k]):
                        if item != item_True:
                            d[k].append((item_True, item))
            failed_nnOut_userOut: Union[Dict[str, List[str], None]] = str(d)

        # tknLbl_Pass != entityLbl_Pass, tknLbl_Pass != userOut_Pass
        entityLbls_status: str = ""
        userOut_status: str = ""
        if tknLbl_Pass != entityLbl_Pass:
            if entityLbl_Pass:
                count["failedTurnsTknLbls_entityLblsPass"] += 1
                entityLbls_status = "+- entityLbls passed  "
            else:
                count["passedTurnsTknLbls_entityLblsFail"] += 1
                entityLbls_status = "+- entityLbls failed  "
        if tknLbl_Pass != userOut_Pass:
            if userOut_Pass:
                count["failedTurnsTknLbls_userOutPass"] += 1
                userOut_status = "+- userOut passed  "
            else:
                count["passedTurnsTknLbls_userOutFail"] += 1
                userOut_status = "+- userOut failed  "

        # print to output files
        with passed_file.open(
                'a') as file_passed, failed_nnOut_tknLblIds_file.open(
                    'a') as file_failed:
            with redirect_stdout(file_passed if tknLbl_Pass else file_failed):
                wrapper: textwrap.TextWrapper = textwrap.TextWrapper(
                    width=80, initial_indent="", subsequent_indent=5 * " ")
                for strng in (
                        f"dlg_id, trn_id: {bch['dlgTrnId'][bch_idx]}   "
                        f"{entityLbls_status}{userOut_status}",
                        "userIn:",
                        f"     {(df[(df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userIn']).item()}",
                        "userIn_filtered_wrds_True:",
                        f"     {' '.join(userIn_filtered_wrds_True)}",
                        "userIn_filtered_wrds:",
                        f"     {' '.join(bch['userIn_filtered_wrds'][bch_idx])}",
                        "(nnIn_tkn, tknLbl_True, nnOut_tknLbl):",
                        f"     {' '.join(tknLbls_assoc)}",
                        "(userIn_filtered_entityWrds_True, "
                        "entityLbls_True, "
                        "nnOut_userIn_filtered_entityWrds, "
                        "nnOut_entityLbls):",
                        f"     {' '.join(entityLbl_assoc)}",
                        "prevTrnUserOut_True:",
                        f"     {(df[(df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['prevTrnUserOut']).item()}",
                        "userOut_True:",
                        f"     {userOut_True}",
                        "nnOut_userOut:",
                        f"     {bch_nnOut_userOut[bch_idx]}",
                        "Failed-nnOut_userOut (userOut_True, nnOut_userOut):"
                        if failed_nnOut_userOut is not None else
                        "Failed-nnOut_userOut: None",
                ):
                    print(wrapper.fill(strng))

                if failed_nnOut_userOut is not None:
                    print(wrapper.fill(f"     {failed_nnOut_userOut}"))
                print('\n')


def prepare_metric(
    bch: Dict[str, Any],
    bch_nnOut_tknLblIds: torch.Tensor,
    dataframes_meta: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[str]]]:
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    assert bch_nnOut_tknLblIds.shape[0] == bch['tknLblIds'].shape[0]

    # tknIds between two SEP belong to tknIds of words in
    # bch['userIn_filtered_wrds']
    nnIn_tknIds_idx_beginEnd: torch.Tensor = (
        bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
    assert bch['nnIn_tknIds']['input_ids'].shape[
        0] * 2 == nnIn_tknIds_idx_beginEnd.shape[
            0], "no_history is False but dataset does not have  history"

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
            nnOut_tknLbl_True = dataframes_meta['tknLblId2tknLbl'][
                bch['tknLblIds'][bch_idx, nnIn_tknIds_idx]]
            assert nnOut_tknLbl_True[0] != "T"
            y_true[-1].append(nnOut_tknLbl_True)
            nnOut_tknLbl = dataframes_meta['tknLblId2tknLbl'][
                bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx]]
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
    dataframes_meta: Dict[str, Any],
    count: Dict[str, Union[int, Dict[str, int]]],
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> None:
    # Print
    from sys import stdout
    from pathlib import Path

    stdoutput = Path('/dev/null')

    for out in (stdoutput, test_results):
        with out.open("a") as results_file:
            with redirect_stdout(stdout if out == stdoutput else results_file):
                turns = (f'# of turns = {count["total_turns"]}')
                failed_nnOut_tknLblIds = ('# of failed nnOut_tknLblIds = '
                                          f'{count["failed_tknLbls"]}')
                failedTurns_nnOut_tknLblIds = (
                    '# of turns of failed nnOut_tknLblIds = '
                    f'{count["failedTurns_tknLbls"]}')
                failedTurns_nnOut_entityLbl = (
                    '# of turns of failed nnOut_entityLbl = '
                    f'{count["failedTurns_entityLbls"]}')
                failedTurns_nnOut_userOut = (
                    '# of turns of failed nnOut_userOut = '
                    f'{count["failedTurns_userOut"]}')
                failedTurnsTknLbls_entityLblsPass = (
                    '# of turns where nnOut_tknLbls fail but nnOut_entityLbls '
                    f'pass = {count["failedTurnsTknLbls_entityLblsPass"]}')
                failedTurnsTknLbls_userOutPass = (
                    '# of turns where nnOut_tknLbls fail but nnOut_userOut '
                    f'pass = {count["failedTurnsTknLbls_userOutPass"]}')
                passedTurnsTknLbls_entityLblsFail = (
                    '# of turns where nnOut_tknLbls pass but nnOut_entityLbls '
                    'fail (MUST be 0) = '
                    f'{count["passedTurnsTknLbls_entityLblsFail"]}')
                passedTurnsTknLbls_userOutFail = (
                    '# of turns where nnOut_tknLbls pass but nnOut_userOut '
                    'fail (MUST be 0) = '
                    f'{count["passedTurnsTknLbls_userOutFail"]}')
                for strng in (
                        turns,
                        failed_nnOut_tknLblIds,
                        failedTurns_nnOut_tknLblIds,
                        failedTurns_nnOut_entityLbl,
                        failedTurns_nnOut_userOut,
                        failedTurnsTknLbls_entityLblsPass,
                        failedTurnsTknLbls_userOutPass,
                        passedTurnsTknLbls_entityLblsFail,
                        passedTurnsTknLbls_userOutFail,
                ):
                    print(strng)
                print('')  # empty line

                relevant_keys_of_dataframes_meta = [
                    'bch sizes',
                    '# of dialog-turns in dataframes',
                    'pandas predict-dataframe file location',
                    'predict-set entityWrds not seen in train-set',
                ]
                for k, v in dataframes_meta.items():
                    if k in relevant_keys_of_dataframes_meta:
                        print(k)
                        print(
                            textwrap.fill(f'{v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))

                tknLblsTrue_not_in_predictSet: str = ""
                passed_nnOut_tknLbl: str = ""
                strng = ('\nfailed tknLbls (* tknLbl_True: # of times -> '
                         'nnOut_tknLbl: # of times, ...):')
                print(strng)
                for tknLblId_True in range(
                        len(dataframes_meta['tknLblId2tknLbl'])):
                    tknLbl_True: str = dataframes_meta['tknLblId2tknLbl'][
                        tknLblId_True]
                    if tknLblId_True not in count["failed_tknLbls_perDlg"]:
                        tknLblsTrue_not_in_predictSet = (
                            f"{tknLblsTrue_not_in_predictSet} {tknLbl_True},")
                    elif (len(count["failed_tknLbls_perDlg"][tknLblId_True])
                          == 1) and (tknLblId_True
                                     in count["failed_tknLbls_perDlg"]
                                     [tknLblId_True]):
                        assert (count['failed_tknLbls_perDlg'][tknLblId_True]
                                [tknLblId_True] == dataframes_meta[
                                    'test-set tknLblIds:count'][tknLblId_True])
                        # all predictions are True
                        passed_nnOut_tknLbl = (
                            f"{passed_nnOut_tknLbl} {tknLbl_True}:"
                            f"{count['failed_tknLbls_perDlg'][tknLblId_True][tknLblId_True]},"
                        )
                    else:
                        assert len(
                            count["failed_tknLbls_perDlg"][tknLblId_True])
                        total_num_times = sum([
                            num_times
                            for num_times in count["failed_tknLbls_perDlg"]
                            [tknLblId_True].values()
                        ])
                        assert total_num_times == dataframes_meta[
                            'test-set tknLblIds:count'][tknLblId_True]
                        failed_nnOut_tknLbl = (f"* {tknLbl_True}:"
                                               f"{total_num_times} ->")
                        if tknLblId_True in count["failed_tknLbls_perDlg"][
                                tknLblId_True]:
                            failed_nnOut_tknLbl = (
                                f"{failed_nnOut_tknLbl} {tknLbl_True}:"
                                f"{count['failed_tknLbls_perDlg'][tknLblId_True][tknLblId_True]},"
                            )
                        for nnOut_tknLblId, num_times in count[
                                "failed_tknLbls_perDlg"][tknLblId_True].items(
                                ):
                            if nnOut_tknLblId != tknLblId_True:
                                failed_nnOut_tknLbl = (
                                    f"{failed_nnOut_tknLbl} "
                                    f"{dataframes_meta['tknLblId2tknLbl'][nnOut_tknLblId]}:{num_times},"
                                )
                        print(
                            textwrap.fill(failed_nnOut_tknLbl,
                                          width=80,
                                          initial_indent=0 * " ",
                                          subsequent_indent=5 * " "))
                print('\n100% passed tknLbls:')
                print(
                    textwrap.fill(passed_nnOut_tknLbl,
                                  width=80,
                                  initial_indent=0 * " ",
                                  subsequent_indent=5 * " "))
                print('\ntknLbls in Train-set but not_in Predict-set:')
                print(
                    textwrap.fill(tknLblsTrue_not_in_predictSet,
                                  width=80,
                                  initial_indent=0 * " ",
                                  subsequent_indent=5 * " "))
                print()

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
