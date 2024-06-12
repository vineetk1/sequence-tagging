"""
Vineet Kumar, xyoom.ai

usage:
    python3 aidiff path1-to-files path2-to-files
       path-to-files does not include the file-name; this path should have the
       following two files: failed_nnOut_tknLblIds.txt, passed_file.txt
"""

from sys import argv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import textwrap
import pickle
from ast import literal_eval
from logging import getLogger

logg = getLogger(__name__)

ID0 = "dlg_id, trn_id"
ID1 = "userIn"
ID2 = "userIn_filtered_wrds_True"
ID3 = "(nnIn_tkn, tknLbl_True, nnOut_tknLbl)"
ID4 = "(userIn_filtered_entityWrds_True, entityLbls_True, nnOut_userIn_filtered_entityWrds, nnOut_entityLbls)"
ID5 = "prevTrnUserOut_True"
ID6 = "userOut_True"
ID7 = "nnOut_userOut"
ID8 = "Failed-nnOut_userOut"


def main():
    assert len(argv) == 3, 'python3 aidiff path1-to-files path2-to-files'
    path1 = Path(argv[len(argv) - 2]).resolve(strict=True)
    file1_pass = path1.joinpath('passed_file.txt').resolve(strict=True)
    file1_fail = path1.joinpath('failed_nnOut_tknLblIds.txt')
    path2 = Path(argv[len(argv) - 1]).resolve(strict=True)
    file2_pass = path2.joinpath('passed_file.txt')
    file2_fail = path2.joinpath('failed_nnOut_tknLblIds.txt')
    all_files_data: List[str] = []
    all_files_offsets: Dict[str, Tuple[int, int]] = {}
    with file1_pass.open('r') as f1p, file1_fail.open(
            'r') as f1f, file2_pass.open('r') as f2p, file2_fail.open(
                'r') as f2f:
        all_files_data = f1p.read().splitlines()
        all_files_offsets['f1p'] = (0, len(all_files_data) - 1)
        all_files_data.extend(f1f.read().splitlines())
        all_files_offsets['f1f'] = (all_files_offsets['f1p'][1] + 1,
                                    len(all_files_data) - 1)
        all_files_data.extend(f2p.read().splitlines())
        all_files_offsets['f2p'] = (all_files_offsets['f1f'][1] + 1,
                                    len(all_files_data) - 1)
        all_files_data.extend(f2f.read().splitlines())
        all_files_offsets['f2f'] = (all_files_offsets['f2p'][1] + 1,
                                    len(all_files_data) - 1)

    trn1f_trn2f, trn1f_trn2p, trn1p_trn2f, stat =\
        boundaries_of_dialogs(all_files_data, all_files_offsets)
    a = 1


def boundaries_of_dialogs(
    all_files_data: List[str], all_files_offsets: Dict[str, Tuple[int, int]]
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], Tuple[
        int, int]]]:
    d1: Dict[Tuple[int, int], Tuple[int, int]] = {}
    d2: Dict[Tuple[int, int], Tuple[int, int]] = {}
    trn_start: int = 0
    dlg_id: str = ""
    trn_id: str = ""
    stat: Dict[str, Any] = {}
    for strg_idx, strg in enumerate(all_files_data):
        if strg.startswith(ID0):
            if strg_idx:
                if strg_idx - 1 <= all_files_offsets['f1f'][1]:
                    d1[(dlg_id, trn_id)] = (trn_start, strg_idx - 1)
                elif strg_idx <= all_files_offsets['f2f'][1]:
                    d2[(dlg_id, trn_id)] = (trn_start, strg_idx - 1)
                else:
                    assert False
                trn_start = strg_idx
            # strg = 'dlg_id, trn_id: [35827, 7]'
            start_idx, comma_idx, end_idx = 0, 0, 0
            for i, c in enumerate(strg):
                if c == '[':
                    start_idx = i
                elif start_idx and c == ',':
                    comma_idx = i
                elif c == ']':
                    end_idx = i
            assert (start_idx < comma_idx) and (comma_idx < end_idx)
            dlg_id = strg[start_idx + 1:comma_idx]
            trn_id = strg[comma_idx + 2:end_idx]
            assert dlg_id.isdigit() and trn_id.isdigit()
            assert (not trn_start and not strg_idx) or (trn_start and strg_idx)
    d2[(dlg_id, trn_id)] = (trn_start, strg_idx)
    stat['num_trns_path1'] = len(d1)
    stat['num_trns_path2'] = len(d2)
    stat['num_trn1_noMatch_trn2'] = len(
        set(d1.keys()).difference(set(d2.keys())))
    stat['num_trn2_noMatch_trn1'] = len(
        set(d2.keys()).difference(set(d1.keys())))

    trn1f_trn2f: List[(int, int, int, int)] = []
    trn1f_trn2p: List[(int, int, int, int)] = []
    trn1p_trn2f: List[(int, int, int, int)] = []
    trn1p_trn2p: int = 0
    for k1, v1 in d1.items():
        try:
            v2 = d2[k1]
        except KeyError:
            continue
        trn1_passed, trn2_passed = None, None
        for v in (v1, v2):
            if v[0] >= all_files_offsets['f1p'][0] and v[
                    1] <= all_files_offsets['f1p'][1]:
                assert trn1_passed is None, "turns not allowed in same path1"
                trn1_passed: bool = True
            elif v[0] >= all_files_offsets['f1f'][0] and v[
                    1] <= all_files_offsets['f1f'][1]:
                assert trn1_passed is None, "turns not allowed in same path1"
                trn1_passed: bool = False
            elif v[0] >= all_files_offsets['f2p'][0] and v[
                    1] <= all_files_offsets['f2p'][1]:
                assert trn2_passed is None, "turns not allowed in same path2"
                trn2_passed: bool = True
            elif v[0] >= all_files_offsets['f2f'][0] and v[
                    1] <= all_files_offsets['f2f'][1]:
                assert trn2_passed is None, "turns not allowed in same path2"
                trn2_passed: bool = False
            else:
                assert False

        if not trn1_passed and not trn2_passed:
            trn1f_trn2f.append((v1[0], v1[1], v2[0], v2[1]))
        elif not trn1_passed and trn2_passed:
            trn1f_trn2p.append((v1[0], v1[1], v2[0], v2[1]))
        elif trn1_passed and not trn2_passed:
            trn1p_trn2f.append((v1[0], v1[1], v2[0], v2[1]))
        else:
            assert trn1_passed and trn2_passed
            trn1p_trn2p += 1
    stat["trn1p_trn2p"] = trn1p_trn2p
    stat["trn1f_trn2f"] = len(trn1f_trn2f)
    stat["trn1f_trn2p"] = len(trn1f_trn2p)
    stat["trn1p_trn2f"] = len(trn1p_trn2f)
    return (trn1f_trn2f, trn1f_trn2p, trn1p_trn2f, stat)


if __name__ == '__main__':
    main()
