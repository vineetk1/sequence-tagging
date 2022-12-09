'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
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
                charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                userIn_wrd2Idx = []
            case "-" | ".":   # hypen or period/decimal-point
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
    return userIn_filter_split


def tknLbls2entity_wrds_lbls(
        bch: Dict[str, Any],
        bch_nnOut_tknLblIds: torch.Tensor,
        ids2tknLbls: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
    # purpose of this function is to check that word-labels are generated
    # correctly
    bch_nnOut_entityWrdLbls = []
    bch_userIn_filtered_entityWrds = []
    # tokens between two SEP belong to tokens of bch['userIn_filtered_wrds']
    nnIn_tknIds_beginEnd_idx = (
            bch['nnIn_tknIds']['input_ids'] == 102).nonzero()

    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        entityWrd, entityWrdLbl = None, None
        entityWrds, entityWrdLbls = [], []
        multipleWord_entity = ""
        userIn_filtered_idx = -1
        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            nnOut_tknLbl = ids2tknLbls[bch_nnOut_tknLblIds[
                                       bch_idx, nnIn_tknIds_idx].item()]
            if nnOut_tknLbl[0] != 'T':  # first token of a word
                userIn_filtered_idx += 1
                if nnOut_tknLbl[0] == 'O':
                    if multipleWord_entity:  # previous multipleWord_entity
                        entityWrds.append(multipleWord_entity)
                    multipleWord_entity = ""
                    continue
                if nnOut_tknLbl[-1] == ')':
                    # there MUST be an opening-parenthesis, else error
                    tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                    entityWrd = nnOut_tknLbl[tknLbl_openParen_idx+1: -1]
                    entityWrdLbl = nnOut_tknLbl[2: tknLbl_openParen_idx]
                else:
                    entityWrd = bch['userIn_filtered_wrds'][bch_idx][
                                                            userIn_filtered_idx]
                    entityWrdLbl = nnOut_tknLbl[2:]
                if nnOut_tknLbl[0] == 'B':
                    if multipleWord_entity:  # previous multipleWord_entity
                        entityWrds.append(multipleWord_entity)
                    entityWrdLbls.append(entityWrdLbl)
                    multipleWord_entity = entityWrd
                else:   # nnOut_tknLbl[0] is 'I'
                    multipleWord_entity = f"{multipleWord_entity} {entityWrd}"
        if multipleWord_entity:  # previous multipleWord_entity
            entityWrds.append(multipleWord_entity)
        bch_nnOut_entityWrdLbls.append(entityWrdLbls)
        bch_userIn_filtered_entityWrds.append(entityWrds)
    return bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls


def ASSERT_tknLbls2entity_wrds_lbls(
        bch: Dict[str, Any],
        bch_nnOut_tknLblIds: torch.Tensor,
        ids2tknLbls: List[str], tokenizer) -> Tuple[List[List[str]], List[
                                                             List[str]]]:
    # purpose of this function is to check that word-labels are generated
    # correctly
    bch_nnOut_entityWrdLbls = []
    bch_userIn_filtered_entityWrds = []
    # tokens between two SEP belong to tokens of bch['userIn_filtered_wrds']
    nnIn_tknIds_beginEnd_idx = (
            bch['nnIn_tknIds']['input_ids'] == 102).nonzero()

    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        assert bch_nnOut_tknLblIds[bch_idx].shape[0] == bch['nnIn_tknIds'][
                'input_ids'][bch_idx].shape[0]
        entityWrd, entityWrdLbl = None, None
        entityWrds, entityWrdLbls = [], []
        multipleWord_entity = ""
        userIn_filtered_idx = -1

        assert_entityWrd, assert_entityWrdLbl = None, None
        assert_nnIn_tkns = []
        assert_nnOut_tknLbls = []
        assert_tknLbls_True = []
        for idx in range(bch_nnOut_tknLblIds[bch_idx].shape[0]):
            # this for-loop is for debugging-only
            assert_nnIn_tkns.append(tokenizer.decode(bch['nnIn_tknIds'][
                                    'input_ids'][bch_idx, idx]))
            assert_nnOut_tknLbls.append(ids2tknLbls[bch_nnOut_tknLblIds[
                                           bch_idx, idx].item()])
            assert_tknLbls_True.append(ids2tknLbls[bch['tknLblIds'][
                                                    bch_idx, idx].item()])

        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            nnOut_tknLbl = ids2tknLbls[bch_nnOut_tknLblIds[
                                       bch_idx, nnIn_tknIds_idx].item()]
            assert nnOut_tknLbl[0] == 'O' or nnOut_tknLbl[
                  0] == 'B' or nnOut_tknLbl[0] == 'I' or nnOut_tknLbl[0] == 'T'
            if nnOut_tknLbl[0] != 'T':  # first token of a word
                userIn_filtered_idx += 1
                if nnOut_tknLbl[0] == 'O':
                    if multipleWord_entity:  # previous multipleWord_entity
                        entityWrds.append(multipleWord_entity)
                    multipleWord_entity = ""
                    continue
                if nnOut_tknLbl[-1] == ')':
                    # there MUST be an opening-parenthesis, else error
                    tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                    entityWrd = nnOut_tknLbl[tknLbl_openParen_idx+1: -1]
                    entityWrdLbl = nnOut_tknLbl[2: tknLbl_openParen_idx]
                else:
                    entityWrd = bch['userIn_filtered_wrds'][bch_idx][
                                                            userIn_filtered_idx]
                    entityWrdLbl = nnOut_tknLbl[2:]
                if nnOut_tknLbl[0] == 'B':
                    if multipleWord_entity:  # previous multipleWord_entity
                        entityWrds.append(multipleWord_entity)
                    entityWrdLbls.append(entityWrdLbl)
                    multipleWord_entity = entityWrd
                    assert_entityWrdLbl = entityWrdLbl
                    assert_entityWrd = entityWrd
                else:   # nnOut_tknLbl[0] is 'I'
                    assert multipleWord_entity
                    assert assert_entityWrdLbl == entityWrdLbl
                    assert assert_entityWrd == entityWrd
                    multipleWord_entity = f"{multipleWord_entity} {entityWrd}"
            else:
                # nnOut_tknLbl is "T" if it is a token of word "O"; else it is
                # "T-xxx(yyy)"
                if nnOut_tknLbl != "T":
                    if nnOut_tknLbl[-1] == ')':
                        # error will occur if there is no opening-parenthesis
                        tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                        entityWrd = nnOut_tknLbl[tknLbl_openParen_idx+1: -1]
                        entityWrdLbl = nnOut_tknLbl[2: tknLbl_openParen_idx]
                    else:
                        entityWrd = bch['userIn_filtered_wrds'][bch_idx][
                                                           userIn_filtered_idx]
                        entityWrdLbl = nnOut_tknLbl[2:]
                    assert assert_entityWrdLbl == entityWrdLbl
                    assert assert_entityWrd == entityWrd
        if multipleWord_entity:  # previous multipleWord_entity
            entityWrds.append(multipleWord_entity)
        assert len(entityWrdLbls) == len(entityWrds)
        bch_nnOut_entityWrdLbls.append(entityWrdLbls)
        bch_userIn_filtered_entityWrds.append(entityWrds)
    assert len(bch_userIn_filtered_entityWrds) == len(bch_nnOut_entityWrdLbls)
    return bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls


def userOut_init():
    userOut = {}
    for car_entity_name in syntheticData.groupOf_car_entity_names:
        # dict keys are in same order as syntheticData.groupOf_car_entity_names
        userOut[car_entity_name] = []
    return userOut


def generate_userOut(
        bch_prev_userOut: List[Dict[str, List[str]]],
        bch_userIn_filtered_entityWrds: List[List[str]],
        bch_wordLabels: List[List[str]]) -> List[Dict[str, List[str]]]:
    return len(bch_prev_userOut) * [""]
    assert len(bch_wordLabels) == len(bch_userIn_filtered_entityWrds)
    assert len(bch_wordLabels) == len(bch_prev_userOut)
    # init_userOut = userOut_init(); bch_userOut =
    # [init_userOut for _ in range(len(bch_wordLabels))] Does NOT work
    # because each copy of dict points to same memory location; i.e. writing a
    # value to a key in a dict will write that value to all dicts
    bch_userOut = [userOut_init() for _ in range(len(bch_wordLabels))]

    for bch_idx in range(len(bch_wordLabels)):
        wrdLbl_idx = 0
        while wrdLbl_idx < len(bch_wordLabels[bch_idx]):
            if (wrdLbl := bch_wordLabels[bch_idx][wrdLbl_idx])[0] == "O":
                wrdLbl_idx += 1
                continue

            def _extract_from_wrdLbl(wrdLbl):
                wrdLbl_prefix = wrdLbl[0]     # 'B' or 'I'
                if wrdLbl[-1] == ')':
                    try:
                        wrdLbl_openParen_idx = wrdLbl.index('(')
                    except ValueError:
                        logg.critical(
                          '{wrdLbl} has closing- but not opening-parenthesis')
                        assert False
                    wrdLbl_name = wrdLbl[2: wrdLbl_openParen_idx]
                    wrdLbl_value = wrdLbl[wrdLbl_openParen_idx+1: -1]
                else:
                    wrdLbl_name = wrdLbl[2:]
                    wrdLbl_value = None
                return wrdLbl_prefix, wrdLbl_name, wrdLbl_value
            wrdLbl_prefix, wrdLbl_name, wrdLbl_value = _extract_from_wrdLbl(
                    wrdLbl)

            match wrdLbl_name:
                case wrdLbl_name if (
                        wrdLbl_name in syntheticData.
                        groupOf_car_entity_names_with_str_entity_values):
                    # bool-OR does not make sense, so AND/OR are ignored
                    bch_userOut[bch_idx][wrdLbl_name].append(
                     wrdLbl_value if wrdLbl_value else bch_userIn_filtered_entityWrds[
                               bch_idx][wrdLbl_idx])
                case wrdLbl_name if ((
                        wrdLbl_name in syntheticData.
                        groupOf_car_entity_names_with_floatInt_entity_values)
                        or (wrdLbl_name in syntheticData.
                            groupOf_car_entity_names_with_int_entity_values)):
                    assert wrdLbl_value is None
                    bch_userOut[bch_idx][wrdLbl_name].append(
                                bch_userIn_filtered_entityWrds[bch_idx][wrdLbl_idx])

            wrdLbl_idx += 1

    return bch_userOut


def userOut2history(
        bch_userOut: List[Dict[str, List[str]]]) -> List[List[str]]:
    return len(bch_userOut) * [[""]]
