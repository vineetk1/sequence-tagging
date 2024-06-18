"""
Vineet Kumar, xyoom.ai

usage:
    python3 aidiff path1-to-files path2-to-files
       (1) path-to-files does not include the file-name; this path shouldi
            have the following two files: failed_nnOut_tknLblIds.txt,
            passed_file.txt
       (2) output is at  path1-to-files/aidiff.txt
"""

from sys import argv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import textwrap
from ast import literal_eval
from logging import getLogger

logg = getLogger(__name__)

ID0 = "dlg_id, trn_id: ["

wrapper1: textwrap.TextWrapper = textwrap.TextWrapper(
    width=80, initial_indent="", subsequent_indent="     ")
wrapper2: textwrap.TextWrapper = textwrap.TextWrapper(
    width=80, initial_indent="     ", subsequent_indent="     ")


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
        all_files_data = f1p.readlines()
        all_files_offsets['f1p'] = (0, len(all_files_data) - 1)
        all_files_data.extend(f1f.readlines())
        all_files_offsets['f1f'] = (all_files_offsets['f1p'][1] + 1,
                                    len(all_files_data) - 1)
        all_files_data.extend(f2p.readlines())
        all_files_offsets['f2p'] = (all_files_offsets['f1f'][1] + 1,
                                    len(all_files_data) - 1)
        all_files_data.extend(f2f.readlines())
        all_files_offsets['f2f'] = (all_files_offsets['f2p'][1] + 1,
                                    len(all_files_data) - 1)

    trn1f_trn2f, trn1f_trn2p, trn1p_trn2f, stat =\
        boundaries_of_turns(all_files_data, all_files_offsets)
    aidiff: List[str] = []
    aidiff.append(f"diff\n  {path1}\n  {path2}\n\n")
    for k, v in stat.items():
        aidiff.append(wrapper1.fill(f"{k} = {v}") + "\n")
    aidiff_print(all_files_data, aidiff, path2, trn1f_trn2f, trn1f_trn2p,
                 trn1p_trn2f)


def boundaries_of_turns(
    all_files_data: List[str], all_files_offsets: Dict[str, Tuple[int, int]]
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]],
           List[Tuple[int, int, int, int]], Dict[str, int]]:
    d1: Dict[Tuple[int, int], Tuple[int, int]] = {}
    d2: Dict[Tuple[int, int], Tuple[int, int]] = {}
    trn_start: int = 0
    dlg_id: str = ""
    trn_id: str = ""
    stat: Dict[str, Any] = {}
    path1_noMatch_path2_txt = ("# of turn-ids (dlg_id, trn_id) in path1 that "
                               "do not have the same turn-ids in path2 "
                               "***MUST be 0***")
    path2_noMatch_path1_txt = ("# of turn-ids (dlg_id, trn_id) in path2 that "
                               "do not have the same turn-ids in path1 "
                               "***MUST be 0***")
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
    stat['# of turns in path1'] = len(d1)
    stat['# of turns in path2'] = len(d2)
    stat[path1_noMatch_path2_txt] = len(
        set(d1.keys()).difference(set(d2.keys())))
    stat[path2_noMatch_path1_txt] = len(
        set(d2.keys()).difference(set(d1.keys())))

    trn1p_trn2p_txt = ("# of turns in path1 that pass and their "
                       "comparable turns in path 2 that pass")
    trn1f_trn2f_txt = ("# of turns in path1 that fail and their "
                       "comparable turns in path 2 that fail")
    trn1f_trn2p_txt = ("# of turns in path1 that fail and their "
                       "comparable turns in path 2 that pass")
    trn1p_trn2f_txt = ("# of turns in path1 that pass and their "
                       "comparable turns in path 2 that fail")
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
    stat[trn1p_trn2p_txt] = trn1p_trn2p
    stat[trn1f_trn2f_txt] = len(trn1f_trn2f)
    stat[trn1f_trn2p_txt] = len(trn1f_trn2p)
    stat[trn1p_trn2f_txt] = len(trn1p_trn2f)
    return (trn1f_trn2f, trn1f_trn2p, trn1p_trn2f, stat)


def aidiff_print(all_files_data: List[str], aidiff: List[str], path2: Path,
                 trn1f_trn2f: List[Tuple[int, int, int, int]],
                 trn1f_trn2p: List[Tuple[int, int, int, int]],
                 trn1p_trn2f: List[Tuple[int, int, int, int]]) -> None:
    trn1f_trn2f_txt = ("Diff between path1 turns that failed and "
                       "their comparable path2 turns that failed")
    trn1f_trn2p_txt = ("Diff between path1 turns that failed and "
                       "their comparable path2 turns that passed")
    trn1p_trn2f_txt = ("Diff between path1 turns that passed and "
                       "their comparable path2 turns that failed")
    map = {0: trn1f_trn2f_txt, 1: trn1f_trn2p_txt, 2: trn1p_trn2f_txt}
    for i, trnsIdxs_lst in enumerate([trn1f_trn2f, trn1f_trn2p, trn1p_trn2f]):
        aidiff.append("\n*** " + wrapper1.fill(f"{map[i]}") + " ***\n")
        startIdx_in_aidiff = len(aidiff)
        trn1Trn2_noChange: int = 0
        aidiff_data: List = []
        count: int = 0
        for trnsIdxs in trnsIdxs_lst:
            count += 1
            no_change, aidiff_data = aidiff_trn1Trn2(all_files_data,
                                                     trn0_strt=trnsIdxs[0],
                                                     trn0_end=trnsIdxs[1],
                                                     trn1_strt=trnsIdxs[2],
                                                     trn1_end=trnsIdxs[3])
            trn1Trn2_noChange += (1 if no_change else 0)
            aidiff.extend(aidiff_data)
        no_change_turns_txt = ("     # of diff turns that did not "
                               f"change = {trn1Trn2_noChange}/{count}\n")
        aidiff.insert(startIdx_in_aidiff, no_change_turns_txt)

    aidiff_file = path2.joinpath('aidiff.txt')
    aidiff_file.touch()
    with aidiff_file.open('w') as f:
        f.writelines(aidiff)


def aidiff_trn1Trn2(all_files_data: List[str], trn0_strt: int, trn0_end: int,
                    trn1_strt: int, trn1_end: int) -> Tuple[int, List[str]]:
    assert all_files_data[trn0_strt].startswith(ID0)
    assert all_files_data[trn1_strt].startswith(ID0)

    headings = [
        ID0,
        "(nnIn_tkn, tknLbl_True, nnOut_tknLbl)",
        "(userIn_filtered_entityWrds_True, entityLbls_True,",
        "prevTrnUserOut_True",
        "nnOut_userOut",
        "Failed-nnOut_userOut",
    ]
    aidiff_data: List[str] = []
    no_change: bool = True
    trnIdxs = {
        0: {
            'hdng_strtIdx': trn0_strt,
            'nxt_hdng_strtIdx': None,
            'trn_endIdx': trn0_end
        },
        1: {
            'hdng_strtIdx': trn1_strt,
            'nxt_hdng_strtIdx': None,
            'trn_endIdx': trn1_end
        }
    }

    for nxt_headingsIdx in range(1, len(headings)):
        for trn_num in trnIdxs.keys():
            for all_files_data_idx in range(
                    trnIdxs[trn_num]['hdng_strtIdx'] + 1,
                    trnIdxs[trn_num]['trn_endIdx']):
                if all_files_data[all_files_data_idx].startswith(
                        headings[nxt_headingsIdx]):
                    trnIdxs[trn_num]['nxt_hdng_strtIdx'] = all_files_data_idx
                    break
            assert trnIdxs[trn_num]['nxt_hdng_strtIdx'] is not None

        if nxt_headingsIdx == 1:
            # copy from "dlg_id, trn_id: ["     upto      "(nnIn_tkn, ...)"
            clip_heading_idx = all_files_data[trnIdxs[0]
                                              ['hdng_strtIdx']].index(']')
            aidiff_data.append(
                f"\n{all_files_data[trnIdxs[0]['hdng_strtIdx']][:clip_heading_idx + 1]}\n"
            )
            aidiff_data.extend(
                all_files_data[trnIdxs[0]['hdng_strtIdx'] +
                               1:trnIdxs[0]['nxt_hdng_strtIdx']])
        elif nxt_headingsIdx == 2:
            # diff "(nnIn_tkn, tknLbl_True, nnOut_tknLbl)"
            aidiff_data.append(all_files_data[trnIdxs[0]['hdng_strtIdx']])
            aidiff_data.append(
                wrapper2.fill("diff path1 (+) vs. path 2 (-)") + '\n')
            trnIdxs[0]['hdng_strtIdx'] += 1
            trnIdxs[1]['hdng_strtIdx'] += 1
            no_diff = diff_parenthesis(trnIdxs, all_files_data, aidiff_data)
            no_change = no_change and no_diff
        elif nxt_headingsIdx == 3:
            # diff "(userIn_filtered_entityWrds_True, entityLbls_True, ....)"
            # special case where heading covers 2 lines
            aidiff_data.extend(all_files_data[
                trnIdxs[0]['hdng_strtIdx']:trnIdxs[0]['hdng_strtIdx'] + 2])
            aidiff_data.append(
                wrapper2.fill("diff path1 (+) vs. path 2 (-)") + '\n')
            trnIdxs[0]['hdng_strtIdx'] += 2
            trnIdxs[1]['hdng_strtIdx'] += 2
            no_diff = diff_parenthesis(trnIdxs, all_files_data, aidiff_data)
            no_change = no_change and no_diff
        elif nxt_headingsIdx == 4:
            # copy from "prevTrnUserOut_True"     upto     "nnOut_userOut"
            aidiff_data.extend(all_files_data[
                trnIdxs[0]['hdng_strtIdx']:trnIdxs[0]['nxt_hdng_strtIdx']])
        elif nxt_headingsIdx == 5:
            # diff "nnOut_userOut"
            aidiff_data.append(all_files_data[trnIdxs[0]['hdng_strtIdx']])
            aidiff_data.append(
                wrapper2.fill(
                    "diff path1 vs. path 2 => (path1 - path2) plus (path2 - path1)"
                ) + '\n')
            trnIdxs[0]['hdng_strtIdx'] += 1
            trnIdxs[1]['hdng_strtIdx'] += 1
            no_diff = diff_dict(trnIdxs, all_files_data, aidiff_data)
            no_change = no_change and no_diff
        else:
            assert False

        for trn_num in trnIdxs.keys():
            trnIdxs[trn_num]['hdng_strtIdx'] = trnIdxs[trn_num][
                'nxt_hdng_strtIdx']
            trnIdxs[trn_num]['nxt_hdng_strtIdx'] = None

    if no_change:
        aidiff_data = aidiff_data[0][:-1] + " no change\n"
    return no_change, aidiff_data


def diff_dict(trnIdxs: Dict[str, Dict[str, int]], all_files_data: List[str],
              aidiff_data: List[str]) -> bool:
    lst: List[str] = join_strings(trnIdxs, all_files_data)
    assert len(lst) == 2
    if not len(lst[0]) or not len(lst[1]):
        aidiff_data.append(wrapper2.fill("atleast one string is empty") + '\n')
        return False
    if lst[0] == lst[1]:
        aidiff_data.append(wrapper2.fill("no difference") + '\n')
        return True

    ld: List[Dict[str, List[str]]] = [
        literal_eval(lst[stng_num]) for stng_num in range(len(lst))
    ]
    dict_diff: str = "{"
    for k in ld[0].keys():
        list_diff = set(ld[0][k]) ^ set(ld[1][k])
        dict_diff = f"{dict_diff}{', ' if dict_diff != '{' else ''}{k}: [{'' if list_diff == set() else list_diff}]"

    aidiff_data.append(wrapper2.fill(dict_diff + "}") + '\n')
    return False


def join_strings(trnIdxs: Dict[str, Dict[str, int]],
                 all_files_data: List[str]) -> List[str]:
    # do not join the heading at the end
    # from a string, strip the initial spaces and the \n at the end
    stngs: List[str] = []
    for trn_num in trnIdxs.keys():
        tmp: str = ""
        for stng_num in range(trnIdxs[trn_num]['hdng_strtIdx'],
                              trnIdxs[trn_num]['nxt_hdng_strtIdx']):
            tmp = (
                f'{tmp}{" " if tmp else ""}{all_files_data[stng_num].strip()}')
        stngs.append(tmp)
    return stngs


def diff_parenthesis(trnIdxs: Dict[str, Dict[str,
                                             int]], all_files_data: List[str],
                     aidiff_data: List[str]) -> bool:
    lst: List[str] = join_strings(trnIdxs, all_files_data)
    assert len(lst) == 2
    if not len(lst[0]) or not len(lst[1]):
        aidiff_data.append(wrapper2.fill("atleast one string is empty") + '\n')
        return False
    if lst[0] == lst[1]:
        aidiff_data.append(wrapper2.fill("no difference") + '\n')
        return True

    paren_pairs: List[List[Tuple[int, int]]] = find_paren_pairs(lst)
    paren_pairs = remove_inner_paren_pairs(paren_pairs)
    diff_paren(lst, paren_pairs, aidiff_data)
    return False


def find_paren_pairs(lst: List[str]) -> List[List[Tuple[int, int]]]:
    stack: List[int] = []
    paren_pairs: List[List[Tuple[int, int]]] = []

    for strng_num, strng in enumerate(lst):
        paren_pairs.append([])
        for idx, char in enumerate(strng):
            if char == '(':
                stack.append(idx)
            elif char == ')':
                i = stack.pop()
                paren_pairs[strng_num].append((i, idx))
        assert not stack
    return paren_pairs


def remove_inner_paren_pairs(
        paren_pairs: List[List[Tuple[int,
                                     int]]]) -> List[List[Tuple[int, int]]]:
    stack: List[List[Tuple[int, int]]] = []

    for strng_num in range(len(paren_pairs)):
        stack.append([])
        for paren_pair in paren_pairs[strng_num]:
            while stack[strng_num] and paren_pair[0] < stack[strng_num][-1][0]:
                assert paren_pair[1] > stack[strng_num][-1][1]
                del stack[strng_num][-1]
            stack[strng_num].append(paren_pair)
    return stack


def diff_paren(lst: List[str], paren_pairs: List[List[Tuple[int, int]]],
               aidiff_data: List[str]):
    assert len(lst) == 2
    if len(paren_pairs[0]) != len(paren_pairs[1]):
        aidiff_data.append(
            wrapper2.fill("unequal # of items to compare; did not compare") +
            '\n')
        aidiff_data.append(wrapper2.fill(f"{lst[0]}") + '\n\n')
        aidiff_data.append(wrapper2.fill(f"{lst[1]}") + '\n')
        return

    for pp0, pp1 in zip(paren_pairs[0], paren_pairs[1]):
        if lst[0][pp0[0]:pp0[1] + 1] != lst[1][pp1[0]:pp1[1] + 1]:
            aidiff_data.append(
                wrapper2.fill('+' + (lst[0][pp0[0]:pp0[1] + 1])) + '\n')
            aidiff_data.append(
                wrapper2.fill('-' + (lst[1][pp1[0]:pp1[1] + 1])) + '\n')


if __name__ == '__main__':
    main()
