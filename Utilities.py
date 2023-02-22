'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Union
import string
from enum import Enum
import torch
import data.synthetic_dataset as syntheticData

logg = getLogger(__name__)


def userIn_filter_splitWords(userIn: str) -> List[str]:
    class CharPos_inUserInWrd(Enum):
        BEGIN = 1           # position of char at beginning of word
        BEGIN2END = 2       # a single char in a word
        MID = 3             # position of char at middle of word
        MID2END = 4         # position of char from middle to end of word
        MID_BEGIN = 5       # position of char at beginning segment of word
        MID_BEGIN2END = 6   # position of char from mid-begin to end of word
    charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
    userIn_wrd2Idx: List[int] = []
    userIn2idx: List[int] = []
    second_pass: List[int] = []

    for char_idx, char in enumerate(userIn):
        if char == " ":
            assert charPos_inUserInWrd == CharPos_inUserInWrd.BEGIN
            assert not userIn_wrd2Idx
            continue
        if char_idx == (len(userIn) - 1) or userIn[char_idx+1] == " ":
            match charPos_inUserInWrd:
                case CharPos_inUserInWrd.BEGIN:
                    charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN2END
                case CharPos_inUserInWrd.MID:
                    charPos_inUserInWrd = CharPos_inUserInWrd.MID2END
                case CharPos_inUserInWrd.MID_BEGIN:
                    charPos_inUserInWrd = CharPos_inUserInWrd.MID_BEGIN2END
                case _:
                    assert False
        assert (not userIn_wrd2Idx) if (charPos_inUserInWrd ==
                                        CharPos_inUserInWrd.BEGIN or
                                        charPos_inUserInWrd ==
                                        CharPos_inUserInWrd.BEGIN2END) else (
                                                                userIn_wrd2Idx)

        match char:
            case char if char not in string.punctuation and char not in '\t\n':
                # begin (include) mid;  begin2end (include + new word) begin;
                # mid (include) mid;  mid2end (include + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (include + new word) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.MID:
                        pass
                    case (CharPos_inUserInWrd.BEGIN |
                          CharPos_inUserInWrd.MID_BEGIN):
                        userIn_wrd2Idx.append(char_idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx)
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        # cannot do userIn_wrd2Idx.clear() because userIn2idx
                        # has reference to userIn_wrd2Idx
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.BEGIN2END:
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn_wrd2Idx.extend([char_idx, char_idx])
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case _:
                        assert False
            case ",":   # comma
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude) mid_begin;
                # mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin;
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID:
                        userIn_wrd2Idx.append(char_idx-1)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID_BEGIN
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID_BEGIN:
                        pass
                    case (CharPos_inUserInWrd.BEGIN |
                          CharPos_inUserInWrd.BEGIN2END):
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case _:
                        assert False
            case "-":   # hypen
                # * (Include as new word; resolve in second pass) begin
                # hyphen is resolved in second pass
                # begin (include + 1 word) begin;
                # begin2end (include + 1 word) begin;
                # mid (include + 2 words) begin;
                # mid2end (include + 2 words) begin;
                # mid_begin (include + 2 words) mid_begin;
                # mid_begin2end (include + 2 words) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.BEGIN:
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.BEGIN2END:
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.MID:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.MID_BEGIN:
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                    case _:
                        assert False
                charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                userIn_wrd2Idx = []
                second_pass.append(len(userIn2idx) - 1)
            case ".":   # period/decimal-point
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (include) mid;  mid2end (exclude + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (exclude + new word) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.MID:
                        pass
                    case CharPos_inUserInWrd.MID_BEGIN:
                        userIn_wrd2Idx.append(char_idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID
                    case CharPos_inUserInWrd.BEGIN:
                        pass
                    case CharPos_inUserInWrd.BEGIN2END:
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case _:
                        assert False
            case "$":   # dollar
                # begin (include + 2 new words) begin;
                # begin2end (include + new word) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.BEGIN:
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.BEGIN2END:
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID:
                        userIn_wrd2Idx.append(char_idx-1)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID_BEGIN
                    case CharPos_inUserInWrd.MID_BEGIN:
                        pass
                    case _:
                        assert False
            case "%":   # percent
                # begin (exclude) begin; begin2end (include + new word) begin;
                # mid (exclude) mid_begin;
                # mid2end (include + 2 new words) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (include + 2 two words) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.BEGIN2END:
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID:
                        userIn_wrd2Idx.append(char_idx-1)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID_BEGIN
                    case CharPos_inUserInWrd.MID_BEGIN:
                        pass
                    case CharPos_inUserInWrd.BEGIN:
                        pass
                    case _:
                        assert False
            case _:
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID_BEGIN2END:
                        userIn2idx.append(userIn_wrd2Idx)
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
                    case CharPos_inUserInWrd.MID:
                        userIn_wrd2Idx.append(char_idx-1)
                        charPos_inUserInWrd = CharPos_inUserInWrd.MID_BEGIN
                    case CharPos_inUserInWrd.MID_BEGIN:
                        pass
                    case CharPos_inUserInWrd.BEGIN:
                        pass
                    case CharPos_inUserInWrd.BEGIN2END:
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                    case _:
                        assert False

    assert charPos_inUserInWrd == CharPos_inUserInWrd.BEGIN
    assert not userIn_wrd2Idx
    for userIn_wrd2Idx in userIn2idx:
        assert (len(userIn_wrd2Idx) % 2) == 0

    userIn_filter_split = []
    for userIn_wrd2Idx in userIn2idx:
        match len(userIn_wrd2Idx):
            case 2:
                userIn_filter_split.append(userIn[userIn_wrd2Idx[
                                                     0]: userIn_wrd2Idx[1]+1])
            case 4:
                word = f'{userIn[userIn_wrd2Idx[0]: userIn_wrd2Idx[1]+1]}{userIn[userIn_wrd2Idx[2]: userIn_wrd2Idx[3]+1]}'
                userIn_filter_split.append(word)
            case _:
                word = ""
                for idx in range(0, len(userIn_wrd2Idx), 2):
                    word = f'{word}{userIn[userIn_wrd2Idx[idx]: userIn_wrd2Idx[idx+1]+1]}'
                userIn_filter_split.append(word)
    assert len(userIn2idx) == len(userIn_filter_split)

    if second_pass:
        remove_word_idxs = []
        for idx_in_userIn2idx in second_pass:
            match userIn_filter_split[idx_in_userIn2idx]:
                case "-":   # hyphen
                    try:
                        if idx_in_userIn2idx:
                            float(userIn_filter_split[idx_in_userIn2idx-1])
                        else:
                            remove_word_idxs.append(idx_in_userIn2idx)
                            continue
                        if idx_in_userIn2idx < len(userIn_filter_split) - 1:
                            if userIn_filter_split[idx_in_userIn2idx+1] == "$":
                                pass
                            else:
                                float(userIn_filter_split[idx_in_userIn2idx+1])
                        else:
                            remove_word_idxs.append(idx_in_userIn2idx)
                            continue
                    except ValueError:
                        remove_word_idxs.append(idx_in_userIn2idx)
                case _:
                    assert False
                    pass
        # for i, prev_word_idx in enumerate(remove_word_idxs):
        #    del userIn_filter_split[prev_word_idx - i]
        # deleting an item changes the indices; delete from high index to low
        for item_idx in reversed(remove_word_idxs):
            del userIn_filter_split[item_idx]
    return userIn_filter_split


def tknLblIds2entity_wrds_lbls(
        bch_nnIn_tknIds: torch.Tensor,
        bch_map_tknIdx2wrdIdx: List[List[int]],
        bch_userIn_filtered: List[List[str]],
        bch_nnOut_tknLblIds: torch.Tensor,
        id2tknLbl: List[str],
        DEBUG_bch_tknLblIds_True,
        DEBUG_tokenizer,
        ) -> Tuple[List[Union[List[str], None]], List[Union[List[str], None]]]:

    # ***************remove DEBUG code starting from here*********************
    D_bch_associate = []
    D_nnIn_tknIds_idx_beginEnd: torch.Tensor = (bch_nnIn_tknIds == 102).nonzero()
    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        D_bch_associate.append([])
        for D_nnIn_tknIds_idx in range(
                (D_nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1), (
                   D_nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
            D_nnIn_tkn = DEBUG_tokenizer.convert_ids_to_tokens(bch_nnIn_tknIds[bch_idx][D_nnIn_tknIds_idx].item())
            D_tknLbl_True = id2tknLbl[DEBUG_bch_tknLblIds_True[bch_idx, D_nnIn_tknIds_idx]]
            D_nnOut_tknLbl = id2tknLbl[bch_nnOut_tknLblIds[bch_idx, D_nnIn_tknIds_idx]]
            D_userIn_filtered_wrd = bch_userIn_filtered[bch_idx][bch_map_tknIdx2wrdIdx[bch_idx][D_nnIn_tknIds_idx]]
            D_bch_associate[-1].append((D_nnIn_tknIds_idx, D_userIn_filtered_wrd, D_nnIn_tkn, D_tknLbl_True, D_nnOut_tknLbl))
    # ******************remove DEBUG code ending  here**********************
    # NOTE: Remove all ASSERTS from Production code of this function
    assert bch_nnOut_tknLblIds.shape == bch_nnIn_tknIds.shape

    bch_nnOut_entityWrdLbls: List[List[str]] = []
    bch_userIn_filtered_entityWrds: List[List[str]] = []
    entityWrd: str = None
    entityWrdLbl: str = None
    userIn_filtered_idx: int = None
    max_count_wrongPredictions_plus1: int = 2 + 1

    # tknIds between two SEP belong to tknIds of words in
    # bch['userIn_filtered']
    nnIn_tknIds_idx_beginEnd: torch.Tensor = (bch_nnIn_tknIds == 102).nonzero()
    tknId_of_O: int = id2tknLbl.index("O")

    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        multipleWord_entity: str = ""
        prev_userIn_filtered_idx: int = None
        prev_BIO: str = None
        bch_nnOut_entityWrdLbls.append([])
        bch_userIn_filtered_entityWrds.append([])
        count_wrongPredictions: int = 0

        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1), (
                   nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
            if (userIn_filtered_idx := bch_map_tknIdx2wrdIdx[bch_idx][
                 nnIn_tknIds_idx]) == prev_userIn_filtered_idx:
                continue    # ignore tknId that is not first-token-of-the-word
            prev_userIn_filtered_idx = userIn_filtered_idx
            if bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx] == tknId_of_O:
                if multipleWord_entity:  # previous multipleWord_entity
                    bch_userIn_filtered_entityWrds[-1].append(
                                                        multipleWord_entity)
                    multipleWord_entity = ""
                prev_BIO = "O"   # next tkn is "B" or "O"
                continue    # ignore tknId of "O"

            nnOut_tknLbl = id2tknLbl[bch_nnOut_tknLblIds[
                                                     bch_idx, nnIn_tknIds_idx]]
            if nnOut_tknLbl[-1] == ')':
                # there MUST be an opening-parenthesis, else error
                tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                entityWrd = nnOut_tknLbl[tknLbl_openParen_idx+1: -1]
                entityWrdLbl = nnOut_tknLbl[2: tknLbl_openParen_idx]
            else:
                entityWrd = bch_userIn_filtered[bch_idx][userIn_filtered_idx]
                entityWrdLbl = nnOut_tknLbl[2:]
            if nnOut_tknLbl[0] == 'B':
                if multipleWord_entity:  # previous multipleWord_entity
                    bch_userIn_filtered_entityWrds[-1].append(
                                                       multipleWord_entity)
                bch_nnOut_entityWrdLbls[-1].append(entityWrdLbl)
                multipleWord_entity = entityWrd
                prev_BIO = "B"    # next tkn is "B" or "I" or "O"
            elif nnOut_tknLbl[0] == 'I':
                if prev_BIO == "B" or prev_BIO == "I":
                    assert multipleWord_entity
                    if entityWrdLbl != bch_nnOut_entityWrdLbls[-1][-1]:
                        count_wrongPredictions = (
                                max_count_wrongPredictions_plus1)
                        break
                        # entityWrdLbl with next-BIO of “I” is different from
                        # entityWrdLbl with prev_BIO of “B”
                        multipleWord_entity = f"WRNG_{entityWrdLbl}-{multipleWord_entity}-{entityWrd}"
                        if ((count_wrongPredictions := count_wrongPredictions +
                           1) >= max_count_wrongPredictions_plus1):
                            break
                    else:
                        multipleWord_entity = f"{multipleWord_entity} {entityWrd}"
                    prev_BIO = "I"    # next tkn is "B" or "I" or "O"
                elif prev_BIO == "O":
                    count_wrongPredictions = max_count_wrongPredictions_plus1
                    break
                    # expected "B" or "O" but model predicts "I"; assume model
                    # is right with prev_BIO but wrong now; so change from "I"
                    # to "B"
                    if multipleWord_entity:  # previous multipleWord_entity
                        bch_userIn_filtered_entityWrds[-1].append(
                                                           multipleWord_entity)
                    bch_nnOut_entityWrdLbls[-1].append(f"Changed_ItoB-{entityWrdLbl}")
                    multipleWord_entity = entityWrd
                    prev_BIO = "B"    # next tkn is "B" or "I" or "O"
                    if ((count_wrongPredictions := count_wrongPredictions + 1)
                            >= max_count_wrongPredictions_plus1):
                        break
                else:   # prev_BIO == None
                    # expected "B" or "O" at start-of-sentence but model
                    # predicts "I"
                    assert prev_BIO is None
                    count_wrongPredictions = max_count_wrongPredictions_plus1
                    break
            else:
                assert not nnOut_tknLbl == 'T'
                count_wrongPredictions = max_count_wrongPredictions_plus1
                break
        if multipleWord_entity:  # previous multipleWord_entity
            bch_userIn_filtered_entityWrds[-1].append(multipleWord_entity)
        if count_wrongPredictions >= max_count_wrongPredictions_plus1:
            del bch_userIn_filtered_entityWrds[-1]
            bch_userIn_filtered_entityWrds.append(None)
            del bch_nnOut_entityWrdLbls[-1]
            bch_nnOut_entityWrdLbls.append(None)
            assert len(bch_userIn_filtered_entityWrds) == len(
                                                    bch_nnOut_entityWrdLbls)
        else:
            assert len(bch_userIn_filtered_entityWrds[bch_idx]) == len(
                                              bch_nnOut_entityWrdLbls[bch_idx])
    assert len(bch_userIn_filtered_entityWrds) == len(bch_nnOut_entityWrdLbls)
    return bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls


def userOut_init():
    userOut = {}
    for car_entityWrdLbl in syntheticData.groupOf_car_entityWrdLbls:
        # dict keys in same order as syntheticData.groupOf_car_entityWrdLbls
        userOut[car_entityWrdLbl] = []
    return userOut


def generate_userOut(
        bch_userOut: List[Dict[str, List[str]]],
        bch_userIn_filtered_entityWrds: List[List[str]],
        bch_entityWrdLbls: List[List[str]]) -> List[Dict[str, List[str]]]:
    assert len(bch_entityWrdLbls) == len(bch_userIn_filtered_entityWrds)
    assert len(bch_entityWrdLbls) == len(bch_userOut)
    # **** For Deployment: Code is written so Asserts, along with associated If
    # statements, can be removed

    for bch_idx in range(len(bch_entityWrdLbls)):
        if bch_entityWrdLbls[bch_idx] is None:
            # ******* do not change bch_userOut[bch_idx] so the user does not
            # see any change on his end; however tell the user that the model
            # cannot understand his text??? *****
            continue
        assert len(bch_entityWrdLbls[bch_idx]) == len(
                                       bch_userIn_filtered_entityWrds[bch_idx])
        wrdLbl_idx = 0
        cmd: str = ""
        unit: str = ""
        carEntityNums: List[str] = []
        carEntityNumsLbl: str = ""
        carEntityNumsNeeded: int = None
        while wrdLbl_idx < len(bch_entityWrdLbls[bch_idx]):
            entityWrd = bch_userIn_filtered_entityWrds[bch_idx][wrdLbl_idx]
            match entityWrdLbl := bch_entityWrdLbls[bch_idx][wrdLbl_idx]:
                case entityWrdLbl if (
                        entityWrdLbl in syntheticData.
                        groupOf_car_entityWrdLbls_with_nonNum_entityWrds):
                    if carEntityNumsLbl:
                        if not carEntityNums:
                            assert False
                        if carEntityNums:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                    cmd, carEntityNumsNeeded, unit = "", None, ""
                    carEntityNums.clear()
                    carEntityNumsLbl = ""
                    bch_userOut[bch_idx][entityWrdLbl].append(entityWrd)
                case entityWrdLbl if ((
                        entityWrdLbl in syntheticData.
                        groupOf_car_entityWrdLbls_with_Num_entityWrds)):
                    if carEntityNumsLbl and carEntityNumsLbl != entityWrdLbl:
                        if not carEntityNums:
                            assert False
                        if carEntityNums:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                        cmd, carEntityNumsNeeded, unit = "", None, ""
                        carEntityNums.clear()
                    carEntityNums.append(entityWrd)
                    carEntityNumsLbl = entityWrdLbl
                case 'units_price' | 'units_mileage':
                    if not unit or (
                            unit in syntheticData.
                            wrds_for_nonCar_entityWrdLbl[entityWrdLbl]):
                        unit = entityWrd
                        if cmd and len(carEntityNums) >= carEntityNumsNeeded:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                            cmd, carEntityNumsNeeded, unit = "", None, ""
                            carEntityNums.clear()
                            carEntityNumsLbl = ""
                    else:
                        # unit not in
                        # syntheticData.wrds_for_nonCar_entityWrdLbl[entityWrdLbl])
                        if not (carEntityNumsLbl and carEntityNums):
                            assert False
                        if carEntityNumsLbl and carEntityNums:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                        # else throw previous collected data
                        cmd, carEntityNumsNeeded = "", None
                        carEntityNums.clear()
                        carEntityNumsLbl = ""
                        unit = entityWrd
                case 'more' | 'less':
                    if cmd:
                        if len(carEntityNums) < carEntityNumsNeeded:
                            assert False
                        if len(carEntityNums) == carEntityNumsNeeded:
                            # start of sentence-segment; less than $5000
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                            carEntityNums.clear()
                            unit, carEntityNumsLbl = "", ""
                            cmd, carEntityNumsNeeded = entityWrdLbl, 1
                        elif len(carEntityNums) > carEntityNumsNeeded:
                            # ASSUME end of sentence-segment; $5000 or less
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums[:-1], carEntityNumsLbl)
                            transition(bch_userOut[bch_idx], entityWrdLbl,
                                       unit, [carEntityNums[-1]],
                                       carEntityNumsLbl)
                            cmd, carEntityNumsNeeded, unit = "", None, ""
                            carEntityNums.clear()
                            carEntityNumsLbl = ""
                        else:   # len(carEntityNums) < carEntityNumsNeeded
                            # bad userIn-seg; throw previous collected info
                            # plus throw new info
                            cmd, carEntityNumsNeeded, unit = "", None, ""
                            carEntityNums.clear()
                            carEntityNumsLbl = ""
                    elif not carEntityNums:
                        # start of sentence-segment; less than $5000
                        carEntityNums.clear()
                        unit, carEntityNumsLbl = "", ""
                        cmd, carEntityNumsNeeded = entityWrdLbl, 1
                    else:   # carEntityNums
                        # ASSUME end of sentence-segment; $5000 or less
                        if len(carEntityNums) > 1:
                            transition(bch_userOut[bch_idx], "", unit,
                                       carEntityNums[:-1], carEntityNumsLbl)
                        transition(bch_userOut[bch_idx], entityWrdLbl,
                                   unit, [carEntityNums[-1]], carEntityNumsLbl)
                        cmd, carEntityNumsNeeded, unit = "", None, ""
                        carEntityNums.clear()
                        carEntityNumsLbl = ""
                case 'range1':
                    if cmd:
                        if len(carEntityNums) < carEntityNumsNeeded:
                            assert False
                        if len(carEntityNums) >= carEntityNumsNeeded:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                        # else throw previous collected data
                    elif carEntityNums:
                        transition(bch_userOut[bch_idx], "", unit,
                                   carEntityNums, carEntityNumsLbl)
                    carEntityNums.clear()
                    unit, carEntityNumsLbl = "", ""
                    cmd, carEntityNumsNeeded = entityWrdLbl, 2
                case 'range2':
                    # $2000 - $4000
                    if cmd:
                        if len(carEntityNums) <= carEntityNumsNeeded:
                            assert False
                        if len(carEntityNums) > carEntityNumsNeeded:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums[:-1], carEntityNumsLbl)
                            cmd, carEntityNumsNeeded = entityWrdLbl, 2
                            carEntityNums = [carEntityNums[-1]]
                        else:   # len(carEntityNums) <= carEntityNumsNeeded
                            # bad userIn-seg; throw previous collected info
                            # plus this one
                            cmd, carEntityNumsNeeded, unit = "", None, ""
                            carEntityNums.clear()
                            carEntityNumsLbl = ""
                    elif len(carEntityNums) == 1:
                        cmd, carEntityNumsNeeded = entityWrdLbl, 2
                    elif len(carEntityNums) > 1:
                        transition(bch_userOut[bch_idx], "", unit,
                                   carEntityNums[:-1], carEntityNumsLbl)
                        cmd, carEntityNumsNeeded = entityWrdLbl, 2
                        carEntityNums = [carEntityNums[-1]]
                    else:   # not carEntityNums:
                        assert False
                        # bad userIn-seg; throw previous collected info
                        # plus this one
                        cmd, carEntityNumsNeeded, unit = "", None, ""
                        carEntityNums.clear()
                        carEntityNumsLbl = ""
                case 'remove':
                    if cmd:
                        if len(carEntityNums) < carEntityNumsNeeded:
                            assert False
                        if len(carEntityNums) >= carEntityNumsNeeded:
                            transition(bch_userOut[bch_idx], cmd, unit,
                                       carEntityNums, carEntityNumsLbl)
                        # else throw previous collected data
                    elif carEntityNums:
                        transition(bch_userOut[bch_idx], "", unit,
                                   carEntityNums, carEntityNumsLbl)
                    cmd, carEntityNumsNeeded, unit = "", None, ""
                    carEntityNums.clear()
                    carEntityNumsLbl = ""

                    wrdLbl_idx += 1
                    if wrdLbl_idx >= len(bch_entityWrdLbls[bch_idx]):
                        break
                    entityWrd = bch_userIn_filtered_entityWrds[
                                                        bch_idx][wrdLbl_idx]
                    match entityWrdLbl := bch_entityWrdLbls[
                                                        bch_idx][wrdLbl_idx]:
                        case 'everything':
                            # remove everything
                            for key in bch_userOut[bch_idx].keys():
                                bch_userOut[bch_idx][key].clear()
                        case entityWrdLbl if (entityWrdLbl in syntheticData.
                                              groupOf_car_entityWrdLbls):
                            # remove brand                 delete brands
                            assert (entityWrd in syntheticData.
                                    groupOf_car_entityWrdLbls) or (
                                            entityWrd[:-1] in syntheticData.
                                            groupOf_car_entityWrdLbls)
                            bch_userOut[bch_idx][entityWrdLbl].clear()
                        case _:
                            assert False
                            wrdLbl_idx -= 1
                case _:
                    assert False
            wrdLbl_idx += 1
        if cmd or len(carEntityNums):
            transition(bch_userOut[bch_idx], cmd, unit, carEntityNums,
                       carEntityNumsLbl)
    return bch_userOut


def transition(userOut: Dict[str, List[str]], cmd: str, unit: str,
               carEntityNums: List[str], carEntityNumsLbl: str) -> None:
    assert carEntityNumsLbl and carEntityNums
    if not carEntityNumsLbl or not carEntityNums:
        return
    assert (carEntityNumsLbl == 'year' or carEntityNumsLbl == 'price' or
            carEntityNumsLbl == 'mileage')
    match cmd:
        case 'more' | 'less':
            userOut[carEntityNumsLbl].append(
             f"{cmd} {carEntityNums[0]}{' ' if unit else ''}{unit}")
            if len(carEntityNums) > 1:
                for carEntityNum in carEntityNums[1:]:
                    userOut[carEntityNumsLbl].append(
                     f"{carEntityNum}{' ' if unit else ''}{unit}")
        case 'range1' | 'range2':
            if len(carEntityNums) < 2:
                assert False
            if len(carEntityNums) == 2:
                # this case avoids a for-loop
                userOut[carEntityNumsLbl].append(
                 f"{carEntityNums[0]}-{carEntityNums[1]}{' ' if unit else ''}{unit}")
            if len(carEntityNums) > 2:
                userOut[carEntityNumsLbl].append(
                 f"{carEntityNums[0]}-{carEntityNums[1]}{' ' if unit else ''}{unit}")
                for carEntityNum in carEntityNums[2:]:
                    userOut[carEntityNumsLbl].append(
                     f"{carEntityNum}{' ' if unit else ''}{unit}")
        case "":
            if not carEntityNums:
                assert False
            if carEntityNums:
                for carEntityNum in carEntityNums:
                    userOut[carEntityNumsLbl].append(
                     f"{carEntityNum}{' ' if unit else ''}{unit}")
        case _:
            assert False


def userOut2history(userOut: Dict[str, List[str]]) -> List[str]:
    history: List[str] = []
    for values in userOut.values():
        history.extend(values)
    return history
