'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Union, Any, Set
import pathlib
import pickle
import pandas as pd
import gc
import random
from collections import Counter
from itertools import zip_longest
from Utilities import userIn_filter_splitWords
import ast
from Synthetic_dataset import (
    PLACEHOLDER_ID_START,
    PLACEHOLDER_ID_END,
    hyphen,
    carEntityNonNumLbls,
    unitsLbls,
    cmdsLbls,
    synonyms_for_carEntityNonNumLbls,
    entityLbls_for_numEntityWrds_mapTo_func,
    entityLbls_for_numEntityWrds_mapTo_genFunc,
)

logg = getLogger(__name__)


class Fill_entityWrds():

    def __init__(self, dataframes_dirPath: str):
        self.nonNumEntityWrds_per_entityLbl: Dict[str, Dict[str, Union[
            Set[Union[str, Tuple[str, str]]], int, bool]]] = {}
        carEntityNonNumLbls_plus_style = list(carEntityNonNumLbls) + ['style']
        synonyms_for_carEntityNonNumLbls_plus_style = (
            synonyms_for_carEntityNonNumLbls)
        synonyms_for_carEntityNonNumLbls_plus_style['style'] = [
            "body_style", "body_styles"
        ]
        for entityLbl in (tuple(unitsLbls.keys()) + tuple(cmdsLbls.keys()) +
                          tuple(carEntityNonNumLbls_plus_style)):
            if entityLbl in carEntityNonNumLbls_plus_style:
                entityWrds = set()
            elif entityLbl in unitsLbls.keys():
                entityWrds = set(unitsLbls[entityLbl])
            elif entityLbl in cmdsLbls.keys():
                entityWrds = set(cmdsLbls[entityLbl])
            else:
                assert False
            self.nonNumEntityWrds_per_entityLbl[entityLbl] = {
                'nonNumEntityWrds': entityWrds,
                'nonNumEntityWrds_idx': 0,
                'all_nonNumEntityWrds_used': False,
            }

        def get_df(car_datasets_dirPath: pathlib.Path) -> pd.DataFrame:
            for child in car_datasets_dirPath.iterdir():  # github, kaggle
                if child.is_dir():  # github
                    for grandchild in child.iterdir(
                    ):  # us-car-models-data-master
                        if grandchild.is_dir():  # us-car-models-data-master
                            for great_grandchild in grandchild.iterdir(
                            ):  # 1992.csv, 1993.csv,
                                if great_grandchild.is_file():  # 1992.csv
                                    if great_grandchild.suffix == '.csv':
                                        yield pd.read_csv(
                                            great_grandchild,
                                            encoding='unicode_escape')

        for df in get_df(
                pathlib.Path(__file__).parent.joinpath('datasets').resolve()):
            for entityLbl in self.nonNumEntityWrds_per_entityLbl:
                if entityLbl in carEntityNonNumLbls_plus_style:
                    for lbl in synonyms_for_carEntityNonNumLbls_plus_style[
                            entityLbl]:
                        if lbl in df.columns:
                            self.nonNumEntityWrds_per_entityLbl[entityLbl][
                                'nonNumEntityWrds'].update(
                                    set(df[lbl].str.lower().unique()))
                            break

        self.nonNumEntityWrds_per_entityLbl['style']['nonNumEntityWrds']: Set[
            str] = {
                entityWrd
                for entityWrds_lst_str in self.
                nonNumEntityWrds_per_entityLbl['style']['nonNumEntityWrds']
                for entityWrd in ast.literal_eval(entityWrds_lst_str)
            }
        self.nonNumEntityWrds_per_entityLbl['style'][
            'nonNumEntityWrds'].update({"van", "minivan"})
        # move entityWrds of 'style' to 'model', and get rid of style
        self.nonNumEntityWrds_per_entityLbl['model']['nonNumEntityWrds'].update(
            self.nonNumEntityWrds_per_entityLbl['style']['nonNumEntityWrds'])
        del self.nonNumEntityWrds_per_entityLbl['style']
        del carEntityNonNumLbls_plus_style
        del synonyms_for_carEntityNonNumLbls_plus_style

        self.multilabel_entityWrds = find_multilabel_entityWrds(
            self.nonNumEntityWrds_per_entityLbl)

        self.nonNumEntityWrds_per_entityLbl = add_typos(
            self.nonNumEntityWrds_per_entityLbl)

        self.nonNumEntityWrds_per_entityLbl = add_spelling_mistakes(
            self.nonNumEntityWrds_per_entityLbl)

        new_multilabel_entityWrds = find_multilabel_entityWrds(
            self.nonNumEntityWrds_per_entityLbl)
        # assert self.multilabel_entityWrds == new_multilabel_entityWrds
        diff = {
            k: new_multilabel_entityWrds[k]
            for k in set(new_multilabel_entityWrds) -
            set(self.multilabel_entityWrds)
        }
        assert not diff

        for entityLbl in self.nonNumEntityWrds_per_entityLbl:
            self.nonNumEntityWrds_per_entityLbl[entityLbl][
                'nonNumEntityWrds'] = list(
                    self.nonNumEntityWrds_per_entityLbl[entityLbl]
                    ['nonNumEntityWrds'])
        dataframes_dirPath = pathlib.Path(dataframes_dirPath).resolve(
            strict=True)
        entityWrds_for_programmer_io_file = dataframes_dirPath.joinpath(
            'entityWrds_for_programmer_io')
        # overwrite entityWrds_for_programmer_io file if it already exists
        with entityWrds_for_programmer_io_file.open('wb') as file:
            pickle.dump(self.nonNumEntityWrds_per_entityLbl,
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL)

        self.entityLbls_of_numEntityWrds_mapTo_genFuncs: Dict[str,
                                                              Dict[str,
                                                                   Any]] = {}
        for entityLbl in entityLbls_for_numEntityWrds_mapTo_genFunc:
            gen_func = entityLbls_for_numEntityWrds_mapTo_genFunc[entityLbl]()
            self.entityLbls_of_numEntityWrds_mapTo_genFuncs[entityLbl] = {
                "gen_func": gen_func,
                'all_numEntityWrds_used': False,
            }

        gc.collect()

    def sentence_label(
        self,
        sentenceWith_placeholders: str,
        tknLblId2tknLbl: List[str],
    ) -> Tuple[str, List[str], List[str]]:
        wrds_wrdLbls: List[Union[str, Dict[str, str]]] = self._fill_entityWrds(
            sentenceWith_placeholders)
        userIn, filtered_wrds_wrdLbls = self._filter_wrdsWrdLbls(wrds_wrdLbls)
        userIn_filtered_wrds, wrdLbls = self._separate_wrds_wrdLbls(
            filtered_wrds_wrdLbls=filtered_wrds_wrdLbls,
            userIn=userIn,
            tknLblId2tknLbl=tknLblId2tknLbl)
        return (userIn, userIn_filtered_wrds, wrdLbls)

    def _fill_entityWrds(
            self, sentenceWith_placeholders: str
    ) -> List[Union[str, Dict[str, str]]]:
        sentenceWith_placeholders: List[str] = sentenceWith_placeholders.split(
        )
        wrds_wrdLbls: List[Union[str, Dict[str, str]]] = []
        for strng in sentenceWith_placeholders:
            if PLACEHOLDER_ID_START not in strng:
                wrds_wrdLbls.append(strng)
                continue
            assert strng[0] == PLACEHOLDER_ID_START
            assert strng.count(PLACEHOLDER_ID_START) == 2
            assert strng[-1] == PLACEHOLDER_ID_END
            assert strng.count(PLACEHOLDER_ID_END) == 2
            try:
                id_end = strng.index(PLACEHOLDER_ID_END)
                assert strng[id_end + 1] == PLACEHOLDER_ID_START
                assert strng[-1] == PLACEHOLDER_ID_END
                lbl = strng[1:id_end]
                wrds = strng[id_end + 2:-1]
            except ValueError:
                assert False
            if (not wrds) and lbl in self.nonNumEntityWrds_per_entityLbl:
                wrds = self.nonNumEntityWrds_per_entityLbl[lbl][
                    'nonNumEntityWrds'][self.nonNumEntityWrds_per_entityLbl[
                        lbl]['nonNumEntityWrds_idx']]
                if self.nonNumEntityWrds_per_entityLbl[lbl][
                        'nonNumEntityWrds_idx'] == (
                            len(self.nonNumEntityWrds_per_entityLbl[lbl]
                                ['nonNumEntityWrds']) - 1):
                    self.nonNumEntityWrds_per_entityLbl[lbl][
                        'nonNumEntityWrds_idx'] = 0
                    self.nonNumEntityWrds_per_entityLbl[lbl][
                        'all_nonNumEntityWrds_used'] = True
                    random.shuffle(self.nonNumEntityWrds_per_entityLbl[lbl]
                                   ['nonNumEntityWrds'])
                else:
                    self.nonNumEntityWrds_per_entityLbl[lbl][
                        'nonNumEntityWrds_idx'] += 1
            elif (not wrds) and lbl in entityLbls_for_numEntityWrds_mapTo_func:
                wrds = entityLbls_for_numEntityWrds_mapTo_func[lbl]()
            elif (not wrds
                  ) and lbl in self.entityLbls_of_numEntityWrds_mapTo_genFuncs:
                wrds, self.entityLbls_of_numEntityWrds_mapTo_genFuncs[lbl][
                    "all_numEntityWrds_used"] = next(
                        self.entityLbls_of_numEntityWrds_mapTo_genFuncs[lbl]
                        ["gen_func"])
            else:
                assert wrds
                assert (lbl in self.nonNumEntityWrds_per_entityLbl) or (
                    lbl in entityLbls_for_numEntityWrds_mapTo_func) or (
                        lbl in self.entityLbls_of_numEntityWrds_mapTo_genFuncs)

            wrds_wrdLbls.append({"entityLbl": lbl, "entityWrds": wrds})
        return wrds_wrdLbls

    def _filter_wrdsWrdLbls(
        self, wrds_wrdLbls: List[Union[str, Dict[str, str]]]
    ) -> Tuple[str, List[Dict[str, Union[List[str], Tuple[List[str], Union[
            List[str], None]]]]]]:
        userIn: str = ""
        filtered_wrds_wrdLbls: List[Dict[str, Union[List[str],
                                                    Tuple[List[str],
                                                          Union[List[str],
                                                                None]]]]] = []

        def get_wrdLbl(wrds_wrdLbl: Union[str, Dict[str, Any]]) -> str:
            wrdLbl: str
            if isinstance(wrds_wrdLbl, str):
                wrdLbl = "O"
            elif isinstance(wrds_wrdLbl, dict):
                wrdLbl = wrds_wrdLbl["entityLbl"]
            else:
                assert False
            return wrdLbl

        def get_wrds(wrds_wrdLbl: Union[str, Dict[str, Any]]) -> str:
            wrds: str
            if isinstance(wrds_wrdLbl, str):
                wrds = wrds_wrdLbl
            elif isinstance(wrds_wrdLbl, dict):
                if isinstance(wrds_wrdLbl["entityWrds"], str):
                    wrds = wrds_wrdLbl["entityWrds"]
                elif isinstance(wrds_wrdLbl["entityWrds"], Tuple):
                    wrds = wrds_wrdLbl["entityWrds"][0]
            else:
                assert False
            return wrds

        def get_correct_wrds(wrds_wrdLbl: Union[str, Dict[str, Any]]) -> str:
            wrds: str
            if isinstance(wrds_wrdLbl, str):
                wrds = None
            elif isinstance(wrds_wrdLbl, dict):
                if isinstance(wrds_wrdLbl["entityWrds"], str):
                    wrds = None
                elif isinstance(wrds_wrdLbl["entityWrds"], Tuple):
                    wrds = wrds_wrdLbl["entityWrds"][1]
            else:
                assert False
            return wrds

        for idx in range(len(wrds_wrdLbls)):
            wrdLbl: str = get_wrdLbl(wrds_wrdLbls[idx])
            assert wrdLbl
            wrds: str = get_wrds(wrds_wrdLbls[idx])
            assert wrds
            correct_wrds: str = get_correct_wrds(wrds_wrdLbls[idx])
            correct_wrds: Union[List[str], None] = userIn_filter_splitWords(
                correct_wrds) if correct_wrds is not None else None
            userIn = f'{userIn}{" " if userIn else ""}{wrds}'

            if wrds == hyphen and wrdLbl != "O":
                assert correct_wrds is None
                strng: str = ""
                if idx and (hyphen not in get_wrds(wrds_wrdLbls[idx - 1])):
                    if idx - 1 and (hyphen not in get_wrds(
                            wrds_wrdLbls[idx - 2])):
                        strng = f'{strng} {get_wrds(wrds_wrdLbls[idx-2])}'
                    strng = f'{strng} {get_wrds(wrds_wrdLbls[idx-1])}'
                strng = f'{strng} {get_wrds(wrds_wrdLbls[idx])}'
                if (idx < len(wrds_wrdLbls) - 1) and (hyphen not in get_wrds(
                        wrds_wrdLbls[idx + 1])):
                    strng = f'{strng} {get_wrds(wrds_wrdLbls[idx+1])}'
                    if (idx + 1 < len(wrds_wrdLbls) - 1) and (
                            hyphen not in get_wrds(wrds_wrdLbls[idx + 2])):
                        strng = f'{strng} {get_wrds(wrds_wrdLbls[idx+2])}'

                if hyphen in userIn_filter_splitWords(strng):
                    filtered_wrds_wrdLbls.append({
                        "entityLbl":
                        wrdLbl,
                        "entityWrds": (wrds.split(), correct_wrds)
                    })
            else:
                filtered_wrds_wrdLbls.append({
                    "entityLbl":
                    wrdLbl,
                    "entityWrds":
                    (userIn_filter_splitWords(wrds), correct_wrds)
                })
                # include case: wrd = landriver, correct_wrd = "land rover"
                assert (len(filtered_wrds_wrdLbls[-1]["entityWrds"][0]) == len(
                    filtered_wrds_wrdLbls[-1]["entityWrds"][1]) or
                        (len(filtered_wrds_wrdLbls[-1]["entityWrds"][0]) == 1
                         and len(filtered_wrds_wrdLbls[-1]["entityWrds"][1])
                         == 2)) if filtered_wrds_wrdLbls[-1]["entityWrds"][
                             1] is not None else True

        return userIn, filtered_wrds_wrdLbls

    def _separate_wrds_wrdLbls(
            self, filtered_wrds_wrdLbls: List[Dict[str,
                                                   Union[List[str],
                                                         Tuple[List[str],
                                                               Union[List[str],
                                                                     None]]]]],
            userIn: str,
            tknLblId2tknLbl: List[str]) -> Tuple[List[str], List[str]]:
        userIn_filtered_wrds: List[str] = []
        wrdLbls: List[str] = []

        for wrds_wrdLbl in filtered_wrds_wrdLbls:
            lbl: str = wrds_wrdLbl["entityLbl"]
            wrds: List[str] = wrds_wrdLbl["entityWrds"][0]
            correct_wrds: List[str] = wrds_wrdLbl["entityWrds"][1]

            assert len(wrds)
            # include case: wrd = landriver, correct_wrd = "land rover"
            assert (len(wrds) == len(correct_wrds) or
                    (len(wrds) == 1 and len(correct_wrds) == 2)
                    ) if correct_wrds is not None else True

            userIn_filtered_wrds.extend(wrds)
            for idx, (wrd, correct_wrd) in enumerate(
                    zip_longest(wrds, (correct_wrds if correct_wrds is not None
                                       else [None]))):
                if lbl == "O":
                    wrdLbls.append("O")
                elif wrd is None:
                    assert idx
                    assert len(wrds) == 1 and len(correct_wrds) == 2
                    # case: wrd = "landriver", correct_wrd = "land rover"
                    # replace previous entry in wrdLbls with this one
                    wrdLbls[-1] = f'B-{lbl}({" ".join(correct_wrds)})'
                else:
                    assert ((wrd is not None and correct_wrd is not None)
                            if correct_wrds is not None else True)
                    if correct_wrds is None:
                        wrdLbls.append(f'{"B-" if not idx else "I-"}{lbl}')
                    else:
                        if wrd == correct_wrd:
                            wrdLbls.append(f'{"B-" if not idx else "I-"}{lbl}')
                        else:
                            wrdLbls.append(f'{"B-" if not idx else "I-"}'
                                           f'{lbl}({correct_wrd})')
            assert len(userIn_filtered_wrds) == len(wrdLbls)

        for wrdLbl in wrdLbls:
            if wrdLbl not in tknLblId2tknLbl:
                tknLblId2tknLbl.append(wrdLbl)

        # check if userIn_filtered_wrds is correct
        temp_userIn_filtered_wrds: List[str] = userIn_filter_splitWords(userIn)
        assert userIn_filtered_wrds == temp_userIn_filtered_wrds
        assert len(userIn_filtered_wrds) == len(wrdLbls)
        # no need to return the list tknLblId2tknLbl
        return userIn_filtered_wrds, wrdLbls

    def all_entityWrds_used(self):
        all_entityWrds_used: bool = True
        for entityLbl in self.nonNumEntityWrds_per_entityLbl:
            all_entityWrds_used = (
                all_entityWrds_used
                and self.nonNumEntityWrds_per_entityLbl[entityLbl]
                ['all_nonNumEntityWrds_used'])
            if not all_entityWrds_used:
                return False
        for entityLbl in self.entityLbls_of_numEntityWrds_mapTo_genFuncs:
            all_entityWrds_used = (
                all_entityWrds_used
                and self.entityLbls_of_numEntityWrds_mapTo_genFuncs[entityLbl]
                ['all_numEntityWrds_used'])
            if not all_entityWrds_used:
                return False
        return True

    def get_multilabel_entityWrds(self) -> Dict[str, List[str]]:
        return self.multilabel_entityWrds


def find_multilabel_entityWrds(
    nonNumEntityWrds_per_entityLbl: Dict[str,
                                         Dict[str,
                                              Union[Set[Union[str,
                                                              Tuple[str,
                                                                    str]]],
                                                    int, bool]]]
) -> Dict[str, List[str]]:
    # Given: the sets in
    # self.nonNumEntityWrds_per_entityLbl[entityLbl]['nonNumEntityWrds']
    # consist of strings and tuple-of-two-strings,
    # e.g. {"red", ("blua indidot salver", "blue indigo silver"),....};
    # In the tuple, only the first string is used for matching

    filtered_nonNumEntityWrds: Dict[str, List[str]] = {}
    for entityLbl in nonNumEntityWrds_per_entityLbl:
        filtered_nonNumEntityWrds[entityLbl] = []
        for entityWrd in nonNumEntityWrds_per_entityLbl[entityLbl][
                'nonNumEntityWrds']:
            if isinstance(entityWrd, tuple):
                filtered_nonNumEntityWrds[entityLbl].append(entityWrd[0])
            elif isinstance(entityWrd, str):
                filtered_nonNumEntityWrds[entityLbl].append(entityWrd)
            else:
                assert False

    count_entityWrds = Counter()
    for entityLbl in filtered_nonNumEntityWrds:
        for entityWrd in filtered_nonNumEntityWrds[entityLbl]:
            count_entityWrds[entityWrd] += 1
    multicount_entityWrds = [k for k, v in count_entityWrds.items() if v > 1]

    multilabel_entityWrds: Dict[str, List[str]] = {}
    for multicount_entityWrd in multicount_entityWrds:
        multilabel_entityWrds[multicount_entityWrd] = []
        for entityLbl in filtered_nonNumEntityWrds:
            if multicount_entityWrd in filtered_nonNumEntityWrds[entityLbl]:
                cnt = filtered_nonNumEntityWrds[entityLbl].count(
                    multicount_entityWrd)
                multilabel_entityWrds[multicount_entityWrd].extend(cnt *
                                                                   [entityLbl])
    return multilabel_entityWrds


def add_typos(
    nonNumEntityWrds_per_entityLbl: Dict[str, Dict[str, Union[Set[str], int,
                                                              bool]]]
) -> Dict[str, Dict[str, Union[Set[Union[str, Tuple[str, str]]], int, bool]]]:
    keyboard_neighbors_of = {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['r', 'd', 's', 'w'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['n', 'b', 'g', 'y', 'u', 'j'],
        'i': ['o', 'k', 'j', 'u'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l'],
        'q': ['a', 'w'],
        'r': ['e', 'd', 'f', 't'],
        's': ['a', 'z', 'x', 'd', 'e', 'w'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'd', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x'],
    }

    def get_typoWrds(entityWrdsChunks: Set[str]) -> List[str]:

        def typo_wrds_from(wrd: str) -> List[str]:
            typo_wrds: List[str] = []
            if (len(wrd) >= 4):
                for wrd_idx in range(len(wrd)):
                    if wrd[wrd_idx] in keyboard_neighbors_of:
                        for typo_char in keyboard_neighbors_of[wrd[wrd_idx]]:
                            typo_wrds.append((f'{wrd[0: wrd_idx]}'
                                              f'{typo_char}{wrd[wrd_idx+1:]}'))
            return typo_wrds

        typing_error_wrds: Set[str] = set()
        for entityWrdsChunk in entityWrdsChunks:
            wrds: List[str] = entityWrdsChunk.split()
            outerList: List[List[str]] = [typo_wrds_from(wrd) for wrd in wrds]
            max_len_of_innerList: int = max(
                [len(innerList) for innerList in outerList])
            for wrds_idx, innerList in enumerate(outerList):
                innerList.extend(
                    (max_len_of_innerList - len(innerList)) * [wrds[wrds_idx]])
            for innerList in outerList:
                assert len(innerList) == max_len_of_innerList
                random.shuffle(innerList)
            for typoWrdNum_in_innerList in range(max_len_of_innerList):
                wrdsChunk: str = ""
                for innerList_num in range(len(outerList)):
                    wrdsChunk = (
                        f'{wrdsChunk}{" " if wrdsChunk else ""}'
                        f'{outerList[innerList_num][typoWrdNum_in_innerList]}')
                if wrdsChunk not in all_nonNumEntityWrds:
                    all_nonNumEntityWrds.add(wrdsChunk)
                    typing_error_wrds.add((wrdsChunk, entityWrdsChunk))
        return typing_error_wrds

    all_nonNumEntityWrds: Set[str] = set()
    for entityLbl in nonNumEntityWrds_per_entityLbl:
        all_nonNumEntityWrds.update(
            nonNumEntityWrds_per_entityLbl[entityLbl]['nonNumEntityWrds'])

    for entityLbl in nonNumEntityWrds_per_entityLbl:
        typo_wrds = get_typoWrds(
            nonNumEntityWrds_per_entityLbl[entityLbl]['nonNumEntityWrds'])
        duplicate_typo_wrds = set()
        for typo_wrd in typo_wrds:
            unique_typo_wrd: bool = True
            for key in nonNumEntityWrds_per_entityLbl:
                if key != entityLbl and unique_typo_wrd:
                    for wrd in nonNumEntityWrds_per_entityLbl[key][
                            'nonNumEntityWrds']:
                        typo_wrd_mod = typo_wrd[0] if isinstance(
                            typo_wrd, tuple) else typo_wrd
                        wrd_mod = wrd[0] if isinstance(wrd, tuple) else wrd
                        if typo_wrd_mod == wrd_mod:
                            duplicate_typo_wrds.add(typo_wrd)
                            unique_typo_wrd = False
                            break
        typo_wrds -= duplicate_typo_wrds
        nonNumEntityWrds_per_entityLbl[entityLbl]['nonNumEntityWrds'].update(
            typo_wrds)
    return nonNumEntityWrds_per_entityLbl


def add_spelling_mistakes(
    nonNumEntityWrds_per_entityLbl: Dict[str,
                                         Dict[str,
                                              Union[Set[Union[str,
                                                              Tuple[str,
                                                                    str]]],
                                                    int, bool]]]
) -> Dict[str, Dict[str, Union[Set[Union[str, Tuple[str, str]]], int, bool]]]:
    spelling_mistakes = {
        'brand': {('mersedes', 'mercedes'), ('mecedes', 'mercedes'),
                  ('Hundai', 'hyundai'), ('Hyndai', 'hyundai'),
                  ('Hyundia', 'hyundai'), ('Huyndai', 'hyundai'),
                  ('Wolksvagen', 'volkswagen'), ('Volkswagon', 'volkswagen'),
                  ('Volckswagen', 'volkswagen'), ('Volxwagen', 'volkswagen'),
                  ('Porche', 'porsche'), ('Porshe', 'porsche'),
                  ('Porsch', 'porsche'), ('Porch', 'porsche'),
                  ('Totota', 'toyota'), ('Toyata', 'toyota'),
                  ('Telsa', 'tesla'), ('Tesl', 'tesla'), ('Tesal', 'tesla'),
                  ('Bently', 'bentley'), ('Benteley', 'bentley'),
                  ('Susuki', 'suzuki'), ('Szuki', 'suzuki'),
                  ('Suzuky', 'suzuki'), ('Suzukey', 'suzuki'),
                  ('Buggati', 'bugatti'), ('Bugati', 'bugatti'),
                  ('Camero', 'camaro'), ('Chevorlet', 'chevrolet'),
                  ('Infinity', 'infiniti'), ('Farrari', 'Ferrari'),
                  ('Ferlary', 'Ferrari'), ('Ferarri', 'Ferrari'),
                  ('MacLaren', 'McLaren'), ('Lemborni', 'Lamborghini'),
                  ('Mayback', 'Maybach'), ('Austin', 'Aston'),
                  ('Suburu', 'Subaru'), ('leksus', 'Lexus'),
                  ('volvi', 'volvo'), ('bwm', 'bmw'), ('auddi', 'audi'),
                  ('landriver', 'Land rover')},
        'model': {('Cyclone', 'Syclone'), ('Sorrento', 'Sorento'),
                  ('Aztec', 'Aztek'), ('Stringer', 'Stinger'),
                  ('Hurricane', 'Huracan'), ('Boxer', 'Boxster'),
                  ('Corrola', 'Corolla'), ('Romero', 'Romeo')}
    }
    for k in spelling_mistakes:
        spell_m = set()
        for tup_wrds in spelling_mistakes[k]:
            spell_m.add((tup_wrds[0].lower(), tup_wrds[1].lower()))
        spelling_mistakes[k] = spell_m

    for entityLbl in nonNumEntityWrds_per_entityLbl:
        if entityLbl in spelling_mistakes:
            duplicate_spellm = set()
            for spellm in spelling_mistakes[entityLbl]:
                unique_spellm: bool = True
                for key in nonNumEntityWrds_per_entityLbl:
                    if key != entityLbl and unique_spellm:
                        for wrd in nonNumEntityWrds_per_entityLbl[key][
                                'nonNumEntityWrds']:
                            spellm_mod = spellm[0] if isinstance(
                                spellm, tuple) else spellm
                            wrd_mod = wrd[0] if isinstance(wrd, tuple) else wrd
                            if spellm_mod == wrd_mod:
                                duplicate_spellm.add(spellm)
                                unique_spellm = False
                                break
            spelling_mistakes[entityLbl] -= duplicate_spellm
            nonNumEntityWrds_per_entityLbl[entityLbl][
                'nonNumEntityWrds'].update(spelling_mistakes[entityLbl])
    return nonNumEntityWrds_per_entityLbl
