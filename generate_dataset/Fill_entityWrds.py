'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Union, Any, Set
import pathlib
import pickle
import pandas as pd
import random
from collections import Counter
from itertools import zip_longest
from Utilities import userIn_filter_splitWords
import ast
from copy import deepcopy
from Synthetic_dataset import (
    PLACEHOLDER_ID_START,
    PLACEHOLDER_ID_END,
    hyphen,
    carEntityNonNumLbls,
    carEntityNumLbls,
    unitsLbls,
    cmdsLbls,
    synonyms_for_carEntityNonNumLbls,
    noNum_labels,
    all_labels,
    entityWrds_withLbl_other,
)

logg = getLogger(__name__)


class Fill_entityWrds():

    def __init__(self, dataframes_dirPath: str):
        self.get_noNum = generate_nonNumbers(dataframes_dirPath)
        self.get_num = generate_numbers(self.get_noNum.get_model_nums())

    def sentence_label(
        self,
        MAX_WRDS_PER_SENTENCE: int,
        sentenceWith_placeholders: str,
        tknLblId2tknLbl: List[str],
    ) -> Tuple[str, List[str], List[str]]:
        wrds_wrdLbls: List[Union[str, Dict[str, str]]] = self._fill_entityWrds(
            sentenceWith_placeholders, MAX_WRDS_PER_SENTENCE)
        userIn, filtered_wrds_wrdLbls = self._filter_wrdsWrdLbls(wrds_wrdLbls)
        userIn_filtered_wrds, wrdLbls = self._separate_wrds_wrdLbls(
            filtered_wrds_wrdLbls=filtered_wrds_wrdLbls,
            userIn=userIn,
            tknLblId2tknLbl=tknLblId2tknLbl)
        return (userIn, userIn_filtered_wrds, wrdLbls)

    def _fill_entityWrds(
            self, sentenceWith_placeholders: str,
            MAX_WRDS_PER_SENTENCE: int) -> List[Union[str, Dict[str, str]]]:
        sentenceWith_placeholders: List[str] = sentenceWith_placeholders.split(
        )
        wrds_wrdLbls: List[Union[str, Dict[str, str]]] = []
        for strng in sentenceWith_placeholders:
            if len(wrds_wrdLbls) >= MAX_WRDS_PER_SENTENCE:
                break
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
            assert lbl in all_labels, f"unknown label= {lbl}"
            assert (wrds == '' or wrds == '$' or wrds == 'than' or wrds == '-'
                    or wrds == 'year' or wrds == 'dollar' or wrds == 'dollars'
                    or wrds[:3]
                    == '___'), f"unknown wrds in <lbl><wrds> = {wrds}"

            if wrds.startswith("___"):
                if wrds[3:] == "rdmYear":
                    wrds_wrdLbls.append({
                        "entityLbl": lbl,
                        "entityWrds": self.get_num.rdmYear()
                    })
                elif wrds[3:] == "notYear":
                    wrds_wrdLbls.append({
                        "entityLbl":
                        lbl,
                        "entityWrds":
                        self.get_num.get_num('notYear')
                    })
                elif wrds[3:] == "entWrdLblOther":
                    assert lbl == 'other'
                    if random.getrandbits(1):
                        wrds = self.get_num.get_num('other')
                    else:
                        wrds = self.get_noNum.get_noNum('entWrdLblOther')
                    wrds_wrdLbls.append({"entityLbl": lbl, "entityWrds": wrds})
                elif wrds[3:] == "multilabel":
                    assert lbl == 'TBD'
                    wrds = self.get_noNum.get_noNum('multilabel')
                    if random.getrandbits(1):
                        wrds_wrdLbls.append({
                            "entityLbl": "other",
                            "entityWrds": wrds[0]
                        })
                        wrds_wrdLbls.append({
                            "entityLbl": wrds[0],
                            "entityWrds": wrds[1]
                        })
                    else:
                        wrds_wrdLbls.append({
                            "entityLbl": wrds[0],
                            "entityWrds": wrds[1]
                        })
                        wrds_wrdLbls.append({
                            "entityLbl": "other",
                            "entityWrds": wrds[0]
                        })
                elif wrds[3:] == "assoc_brand_modelNum":
                    assert lbl == 'TBD'
                    wrds = self.get_noNum.get_noNum('assoc_brand_modelNum')
                    wrds_wrdLbls.append({
                        "entityLbl": 'brand',
                        "entityWrds": wrds[0]
                    })
                    wrds_wrdLbls.append({
                        "entityLbl": 'model',
                        "entityWrds": wrds[1]
                    })
                else:
                    assert False, f"unknown entityWrd: {wrds}"
            elif wrds:
                wrds_wrdLbls.append({"entityLbl": lbl, "entityWrds": wrds})
            elif (not wrds) and (lbl in noNum_labels) and (lbl != 'TBD'):
                wrds_wrdLbls.append({
                    "entityLbl": lbl,
                    "entityWrds": self.get_noNum.get_noNum(lbl)
                })
            elif (not wrds) and lbl == "year":
                wrds_wrdLbls.append({
                    "entityLbl": lbl,
                    "entityWrds": self.get_num.seqYear()
                })
            elif (not wrds) and (lbl == "mileage" or lbl == "price"):
                wrds_wrdLbls.append({
                    "entityLbl": lbl,
                    "entityWrds": self.get_num.get_num(lbl)
                })
            elif (not wrds) and lbl == "setting":
                wrds_wrdLbls.append({
                    "entityLbl": lbl,
                    "entityWrds": self.get_num.setting()
                })
            else:
                assert False, f"unknown label= {lbl}"
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
            assert isinstance(wrds_wrdLbl, dict)
            if wrds_wrdLbl["entityLbl"] == 'other':
                wrdLbl = "O"
            else:
                wrdLbl = wrds_wrdLbl["entityLbl"]
            return wrdLbl

        def get_wrds(wrds_wrdLbl: Union[str, Dict[str, Any]]) -> str:
            wrds: str
            assert isinstance(wrds_wrdLbl, dict)
            if isinstance(wrds_wrdLbl["entityWrds"], str):
                wrds = wrds_wrdLbl["entityWrds"]
            elif isinstance(wrds_wrdLbl["entityWrds"], Tuple):
                wrds = wrds_wrdLbl["entityWrds"][0]
            else:
                assert False
            return wrds

        def get_correct_wrds(wrds_wrdLbl: Union[str, Dict[str, Any]]) -> str:
            wrds: str
            assert isinstance(wrds_wrdLbl, dict)
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

            if wrds == hyphen:
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

    def all_entityWrds_used(self, NUM_TIMES_ENTITYWRDS_USED: int):
        for class_obj in [self.get_noNum, self.get_num]:
            if not class_obj.all_entityWrds_used(NUM_TIMES_ENTITYWRDS_USED):
                return False
        return True


class generate_nonNumbers():

    def __init__(self, dataframes_dirPath: str):
        # all Sets are converted to Tuples, later in this function
        self.nonNumWrds_per_entityLbl: Dict[str,
                                            Dict[str,
                                                 Union[Set[Union[str,
                                                                 Tuple[str,
                                                                       str]]],
                                                       int, bool]]] = {}

        # fill entityWrds in some entityLbls of
        # self.nonNumWrds_per_entityLbl
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
            self.nonNumWrds_per_entityLbl[entityLbl] = {
                'items': entityWrds,
                'idx': 0,
                'all_used': 0,
            }

        # (1) fill entityWrds of those entityLbls that were not previously
        #     filled;
        # (2) extract row-level info from each df; info such as associating
        #     brands with their model numbers such as "saab" with "9000"
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

        assoc_brand_modelNum: List[Tuple[str, str]] = []
        for df in get_df(
                pathlib.Path(__file__).parent.joinpath('datasets').resolve()):
            # fill entityWrds of entityLbls that were not previously filled
            for entityLbl in self.nonNumWrds_per_entityLbl:
                if entityLbl in carEntityNonNumLbls_plus_style:
                    for lbl in synonyms_for_carEntityNonNumLbls_plus_style[
                            entityLbl]:
                        if lbl in df.columns:
                            self.nonNumWrds_per_entityLbl[entityLbl][
                                'items'].update(
                                    set(df[lbl].str.lower().unique()))
                            break

            # extract any type of info from each row
            for idx in df.index:
                if df['model'][idx].isdecimal():
                    if "brand" in df.columns:
                        assoc_brand_modelNum.append(
                            (df['brand'][idx], df['model'][idx]))
                    elif "make" in df.columns:
                        assoc_brand_modelNum.append(
                            (df['make'][idx], df['model'][idx]))
                    else:
                        assert False, "Neither brand or make in df.columns"
        assoc_brand_modelNum = list(set(assoc_brand_modelNum))

        # (1) model_nums has all number-strings that are in "model";
        # (2) remove number-strings (e.g. "9000", not "9000 127") from model
        # (3) model names like "vf 9" get an additional name of "vf9"
        self.model_nums: List[str] = []
        for strng in list(self.nonNumWrds_per_entityLbl['model']['items']):
            if strng.isdecimal():
                # "9000" => True, "9000 127" => False
                self.nonNumWrds_per_entityLbl['model']['items'].remove(strng)
                if strng not in self.model_nums:
                    self.model_nums.append(strng)
            strng = strng.split()
            if len(strng) == 2 and (
                    not strng[0].isdecimal()) and strng[1].isdecimal():
                # user can write "vf 9" also as "vf9"
                self.nonNumWrds_per_entityLbl['model']['items'].add(
                    f"{strng[0]}{strng[1]}")
        self.model_nums: Tuple[str] = tuple(self.model_nums)

        # assert if brand or color have number-string
        for entityLbl in ("brand", "color"):
            for strng in self.nonNumWrds_per_entityLbl[entityLbl]['items']:
                if strng.isdecimal():
                    assert False, "number found in brand or color"

        # add names that are not present
        # add 'mercedes' because it is only present as 'mercedes-benz'
        self.nonNumWrds_per_entityLbl['brand']['items'].add('mercedes')

        # convert entityWrds of 'style' from strange format
        self.nonNumWrds_per_entityLbl['style']['items']: Set[str] = {
            entityWrd
            for entityWrds_lst_str in self.nonNumWrds_per_entityLbl['style']
            ['items'] for entityWrd in ast.literal_eval(entityWrds_lst_str)
        }
        self.nonNumWrds_per_entityLbl['style']['items'].update(
            {"van", "minivan"})

        # move entityWrds of 'style' to 'model', and get rid of style
        self.nonNumWrds_per_entityLbl['model']['items'].update(
            self.nonNumWrds_per_entityLbl['style']['items'])
        del self.nonNumWrds_per_entityLbl['style']
        del carEntityNonNumLbls_plus_style
        del synonyms_for_carEntityNonNumLbls_plus_style

        # create a set that keep tracks of all the entityWrds used so far; new
        # entityWrds from spelling-mistakes and typos are created only if they
        # do not exist in this set
        all_nonNumEntityWrds: Set[str] = set()
        for entityLbl in self.nonNumWrds_per_entityLbl:
            # all_nonNumEntityWrds.update(
            #     self.nonNumWrds_per_entityLbl[entityLbl]['items'])
            for strg in self.nonNumWrds_per_entityLbl[entityLbl]['items']:
                for wrd in strg.split():
                    try:
                        float(wrd)   # ignore if wrd in a number
                    except ValueError:
                        all_nonNumEntityWrds.add(wrd)   # wrd is not a number

        for entityWrds in synonyms_for_carEntityNonNumLbls.values():
            all_nonNumEntityWrds.update(set(entityWrds))
        all_nonNumEntityWrds.update(set(carEntityNumLbls))

        # (1) generate non-entity-words, called other-words; (2) other-words
        # must not have the words in all_nonNumEntityWrds
        with open("/etc/dictionaries-common/words", "r") as file:
            other_words = file.read()
            other_words = set(map(str.lower, other_words.split()))
        other_words = list(other_words - all_nonNumEntityWrds)
        random.shuffle(other_words)

        # entityWrds that have more than one entityLbl; also remove
        # those entityWrds from those entityLbls
        multilabel_entityWrds_orignal: Dict[
            str, List[str]] = find_multilabel_entityWrds(
                self.nonNumWrds_per_entityLbl)
        for entityWrd, entityLbls in multilabel_entityWrds_orignal.items():
            for entityLbl in entityLbls:
                self.nonNumWrds_per_entityLbl[entityLbl]['items'].remove(
                    entityWrd)

        # add spelling mistakes to entityWrds
        for entityLbl in self.nonNumWrds_per_entityLbl:
            add_spelling_mistakes(
                entityLbl, self.nonNumWrds_per_entityLbl[entityLbl]['items'],
                all_nonNumEntityWrds)

        multilabel_entityWrds: Dict[str, List[str]] = deepcopy(
            multilabel_entityWrds_orignal)
        for entityWrd, multilabels in multilabel_entityWrds_orignal.items():
            entityWrd_withSpellMistakes: Set[str] = {entityWrd}
            add_spelling_mistakes(multilabels[0], entityWrd_withSpellMistakes,
                                  all_nonNumEntityWrds)
            for misspelled_entityWrd in entityWrd_withSpellMistakes:
                if misspelled_entityWrd not in multilabel_entityWrds_orignal:
                    multilabel_entityWrds[
                        misspelled_entityWrd] = multilabel_entityWrds_orignal[
                            misspelled_entityWrd[1]]

        # add typos to entityWrds
        for entityLbl in self.nonNumWrds_per_entityLbl:
            typo_wrds: Set[Tuple[str, str]] = get_typos(
                self.nonNumWrds_per_entityLbl[entityLbl]['items'],
                all_nonNumEntityWrds)
            self.nonNumWrds_per_entityLbl[entityLbl]['items'].update(typo_wrds)

        typo_wrds: Set[Tuple[str, str]] = get_typos(
            set(multilabel_entityWrds_orignal.keys()), all_nonNumEntityWrds)
        for typo_wrd in typo_wrds:
            assert typo_wrd[0] not in multilabel_entityWrds.keys()
            multilabel_entityWrds[typo_wrd] = multilabel_entityWrds_orignal[
                typo_wrd[1]]
        # change from Dict[str, List[str]] to List[Tuple[str, str]]
        new_multilabel_entityWrds: List[Tuple[str, str]] = []
        for entityWrd, multilabels in multilabel_entityWrds.items():
            for multilabel in multilabels:
                new_multilabel_entityWrds.append((multilabel, entityWrd))

        # check that no new multilabel entityWrds are generated in
        # self.nonNumWrds_per_entityLbl because all such entityWrds were
        # removed before spelling-mistakes and typo-errors were added
        test_multilabel_entityWrds = find_multilabel_entityWrds(
            self.nonNumWrds_per_entityLbl)
        assert (test_multilabel_entityWrds == {}
                ), f'new multilabel entitywrds = {test_multilabel_entityWrds}'

        # convert to python-list all of
        # self.nonNumWrds_per_entityLbl[entityLbl]['items']
        for entityLbl in self.nonNumWrds_per_entityLbl:
            self.nonNumWrds_per_entityLbl[entityLbl]['items'] = list(
                self.nonNumWrds_per_entityLbl[entityLbl]['items'])

        # add remaining (lbl, wrds) to self.nonNumWrds_per_entityLbl
        # 'other' is the only label; rest are not but they are used here anyway
        lbls_wrds = [('assoc_brand_modelNum', assoc_brand_modelNum),
                     ('multilabel', new_multilabel_entityWrds),
                     ('other', other_words),
                     ('entWrdLblOther', entityWrds_withLbl_other)]
        for lbl_wrd in lbls_wrds:
            self.nonNumWrds_per_entityLbl[lbl_wrd[0]] = {
                'items': lbl_wrd[1],
                'idx': 0,
                'all_used': 0,
            }

        # save data_structures to a file
        data_structures = [self.nonNumWrds_per_entityLbl, self.model_nums]
        dataframes_dirPath = pathlib.Path(dataframes_dirPath).resolve(
            strict=True)
        data_structures_file = dataframes_dirPath.joinpath(
            'data_structures.pickle')
        with data_structures_file.open('wb') as file:
            for data_structure in data_structures:
                pickle.dump(data_structure,
                            file,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def get_model_nums(self) -> Tuple[str]:
        return self.model_nums

    def get_noNum(self, lbl) -> Tuple[Dict[str, str]]:
        # Note the following are not labels but they are used here
        # anyway: entWrdLblOther, multilabel, assoc_brand_modelNum

        # do not use 'entWrdLblOther' because it confuses Bert; use 'other'
        if lbl == 'entWrdLblOther':
            lbl = 'other'
        wrds = self.nonNumWrds_per_entityLbl[lbl]['items'][
            self.nonNumWrds_per_entityLbl[lbl]['idx']]
        if self.nonNumWrds_per_entityLbl[lbl]['idx'] == (
                len(self.nonNumWrds_per_entityLbl[lbl]['items']) - 1):
            self.nonNumWrds_per_entityLbl[lbl]['idx'] = 0
            self.nonNumWrds_per_entityLbl[lbl]['all_used'] += 1
            random.shuffle(self.nonNumWrds_per_entityLbl[lbl]['items'])
        else:
            self.nonNumWrds_per_entityLbl[lbl]['idx'] += 1
        return wrds

    def all_entityWrds_used(self, NUM_TIMES_ENTITYWRDS_USED: int) -> bool:
        for entityLbl in self.nonNumWrds_per_entityLbl:
            if not (entityLbl == 'other' or entityLbl == 'entWrdLblOther'):
                if not (self.nonNumWrds_per_entityLbl[entityLbl]['all_used'] >=
                        NUM_TIMES_ENTITYWRDS_USED):
                    return False
        return True


def find_multilabel_entityWrds(
    nonNumEntityWrds_per_entityLbl: Dict[str,
                                         Dict[str,
                                              Union[Set[Union[str,
                                                              Tuple[str,
                                                                    str]]],
                                                    int, bool]]]
) -> Dict[str, List[str]]:
    # Given: the sets in
    # self.nonNumEntityWrds_per_entityLbl[entityLbl]['items']
    # consist of strings and tuple-of-two-strings,
    # e.g. {"red", ("blua indidot salver", "blue indigo silver"),....};
    # In the tuple, only the first string is used for matching

    filtered_nonNumEntityWrds: Dict[str, List[str]] = {}
    for entityLbl in nonNumEntityWrds_per_entityLbl:
        filtered_nonNumEntityWrds[entityLbl] = []
        for entityWrd in nonNumEntityWrds_per_entityLbl[entityLbl]['items']:
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


def get_typos(entityWrdsChunks: Set[Union[str, Tuple[str, str]]],
              all_nonNumEntityWrds: Set[str]) -> Set[Tuple[str, str]]:
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

    def typo_wrds_from(wrd: str) -> List[str]:
        typo_wrds: List[str] = []
        if (len(wrd) >= 4):
            for wrd_idx in range(len(wrd)):
                if wrd[wrd_idx] in keyboard_neighbors_of:
                    for typo_char in keyboard_neighbors_of[wrd[wrd_idx]]:
                        typo_wrds.append((f'{wrd[0: wrd_idx]}'
                                          f'{typo_char}{wrd[wrd_idx+1:]}'))
        return typo_wrds

    typing_error_wrds: Set[Tuple[str, str]] = set()
    for entityWrdsChunk in entityWrdsChunks:
        if isinstance(entityWrdsChunk, tuple):
            # tuple have spelling mistakes only
            continue
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


def add_spelling_mistakes(entityLbl: str, nonNumEntityWrds: Set[str],
                          all_nonNumEntityWrds: Set[str]):
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
    if entityLbl in spelling_mistakes:
        for spellm in spelling_mistakes[entityLbl]:
            spellm = (spellm[0].lower(), spellm[1].lower())
            if spellm[1] in nonNumEntityWrds and spellm[
                    0] not in all_nonNumEntityWrds:
                all_nonNumEntityWrds.add(spellm[0])
                nonNumEntityWrds.add(spellm)


class generate_numbers():

    def __init__(self, model_nums: List[str]):
        self.year_seq_range = (1970, 2025)
        setting_seq_range = (1, 20)
        self.seq_done = {'year': 0, 'setting': 0}
        self.year_seq = self._sequence_gen(self.year_seq_range[0],
                                           self.year_seq_range[1], 'year')
        self.setting_seq = self._sequence_gen(setting_seq_range[0],
                                              setting_seq_range[1], 'setting')
        self.nums: List[str] = list(model_nums) + [
            str(i)
            for i in range(self.year_seq_range[0], self.year_seq_range[1] + 1)
        ] + [
            str(i)
            for i in range(setting_seq_range[0], setting_seq_range[1] + 1)
        ]
        self.nums_idx = {
            # idx is pointing at an element that has not been retrieved, or
            # there is no element in that location
            'price': 0,
            'mileage': 0,
            'notYear': 0,
            'other': 0
        }

    def all_entityWrds_used(self, NUM_TIMES_ENTITYWRDS_USED: int) -> bool:
        for v in self.seq_done.values():
            if not (v >= NUM_TIMES_ENTITYWRDS_USED):
                return False
        return True

    def rdmYear(self) -> str:
        # get a random int in the Year sequence
        return str(
            random.randint(self.year_seq_range[0], self.year_seq_range[1]))

    def seqYear(self) -> str:
        # get the next int from the Year sequence
        return next(self.year_seq)

    def setting(self) -> str:
        # get the next int from the Setting sequence
        return next(self.setting_seq)

    def get_num(self, lbl: str) -> str:
        # 'notYear' is not a label but it is used here anyway
        # 'entWrdLblOther' has a label of 'other', which is used here
        assert (lbl == 'price' or lbl == 'mileage' or lbl == 'notYear'
                or lbl == 'other'), f"unknown label = {lbl}"
        self._reduce_nums()

        def _rdmInt_notYearSeq():
            for loop in range(10):
                numStr = self._rdmInt()
                if int(numStr) >= self.year_seq_range[0] and int(
                        numStr) <= self.year_seq_range[1]:
                    assert loop < 9,\
                          "cannot find random int that is NOT in Year sequence"
                else:
                    break
            return numStr

        if self.nums_idx[lbl] == len(self.nums):
            # either list is empty or all elements retrieved
            if lbl == 'notYear':
                self.nums.append(_rdmInt_notYearSeq())
            else:
                if random.getrandbits(1):
                    self.nums.append(self._rdmFloat())
                else:
                    self.nums.append(self._rdmInt())
        elif self.nums_idx[lbl] < len(self.nums):
            # not all elements retrieved from list
            if lbl == 'notYear':
                while self.nums_idx[lbl] < len(self.nums):
                    num = float(self.nums[self.nums_idx[lbl]])
                    if num >= self.year_seq_range[
                            0] and num <= self.year_seq_range[1]:
                        self.nums_idx[lbl] += 1
                        continue
                    else:
                        break
                if self.nums_idx[lbl] == len(self.nums):
                    self.nums.append(_rdmInt_notYearSeq())
        else:
            # self.nums_idx[lbl] > len(self.nums)
            assert False
        self.nums_idx[lbl] += 1
        return self.nums[self.nums_idx[lbl] - 1]

    def _rdmInt(self) -> str:
        return str(random.randint(0, 999999))

    def _rdmFloat(self) -> str:
        return str(round(random.uniform(0, 999999), 2))

    def _sequence_gen(
            self, START: int, END: int, entity: str
    ) -> Tuple[str, bool]:  # this generator has infinite loop
        num: int = END
        while True:
            if num < START:
                self.seq_done[entity] += 1
                num = END
            yield str(num)
            num -= 1

    def _reduce_nums(self):
        # reduce size of list by deleting "n" used-up elements from front
        reduction_len = min(self.nums_idx.values())
        if reduction_len >= 10:
            del self.nums[:reduction_len]
            for k, v in self.nums_idx.items():
                self.nums_idx[k] = v - reduction_len
