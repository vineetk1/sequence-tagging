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


def preTokenize_splitWords(strng: str) -> List[str]:
    class CharPos_inStrngWord(Enum):
        BEGIN = 1           # position of char at beginning of word
        BEGIN2END = 2       # a single char in a word
        MID = 3             # position of char at middle of word
        MID2END = 4         # position of char from middle to end of word
        MID_BEGIN = 5       # position of char at beginning segment of word
        MID_BEGIN2END = 6   # position of char from mid-begin to end of word
    charPos_inStrngWord = CharPos_inStrngWord.BEGIN
    strngWord2Idx: List[int] = []
    strng2idx: List[int] = []

    for char_idx, char in enumerate(strng):
        if char == " ":
            assert charPos_inStrngWord == CharPos_inStrngWord.BEGIN
            assert not strngWord2Idx
            continue
        if char_idx == (len(strng) - 1) or strng[char_idx+1] == " ":
            match charPos_inStrngWord:
                case CharPos_inStrngWord.BEGIN:
                    charPos_inStrngWord = CharPos_inStrngWord.BEGIN2END
                case CharPos_inStrngWord.MID:
                    charPos_inStrngWord = CharPos_inStrngWord.MID2END
                case CharPos_inStrngWord.MID_BEGIN:
                    charPos_inStrngWord = CharPos_inStrngWord.MID_BEGIN2END
                case _:
                    assert False
        assert (not strngWord2Idx) if (charPos_inStrngWord ==
                                       CharPos_inStrngWord.BEGIN or
                                       charPos_inStrngWord ==
                                       CharPos_inStrngWord.BEGIN2END) else (
                                                                 strngWord2Idx)

        match char:
            case char if char not in string.punctuation and char not in '\t\n':
                # begin (include) mid;  begin2end (include + new word) begin;
                # mid (include) mid;  mid2end (include + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (include + new word) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID:
                        pass
                    case (CharPos_inStrngWord.BEGIN |
                          CharPos_inStrngWord.MID_BEGIN):
                        strngWord2Idx.append(char_idx)
                        charPos_inStrngWord = CharPos_inStrngWord.MID
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx)
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strngWord2Idx.extend([char_idx, char_idx])
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case _:
                        assert False
            case ",":   # comma
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude) mid_begin;
                # mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin;
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        charPos_inStrngWord = CharPos_inStrngWord.MID_BEGIN
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN:
                        pass
                    case (CharPos_inStrngWord.BEGIN |
                          CharPos_inStrngWord.BEGIN2END):
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case _:
                        assert False
                charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                strngWord2Idx = []
            case "-" | ".":   # hypen or period/decimal-point
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (include) mid;  mid2end (exclude + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (exclude + new word) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID:
                        pass
                    case CharPos_inStrngWord.MID_BEGIN:
                        strngWord2Idx.append(char_idx)
                        charPos_inStrngWord = CharPos_inStrngWord.MID
                    case CharPos_inStrngWord.BEGIN:
                        pass
                    case CharPos_inStrngWord.BEGIN2END:
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case _:
                        assert False
            case "$":   # dollar
                # begin (include + 2 new words) begin;
                # begin2end (include + new word) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.BEGIN:
                        strng2idx.append([char_idx, char_idx])
                    case CharPos_inStrngWord.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        charPos_inStrngWord = CharPos_inStrngWord.MID_BEGIN
                    case CharPos_inStrngWord.MID_BEGIN:
                        pass
                    case _:
                        assert False
            case "%":   # percent
                # begin (exclude) begin; begin2end (include + new word) begin;
                # mid (exclude) mid_begin;
                # mid2end (include + 2 new words) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (include + 2 two words) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        charPos_inStrngWord = CharPos_inStrngWord.MID_BEGIN
                    case CharPos_inStrngWord.MID_BEGIN:
                        pass
                    case CharPos_inStrngWord.BEGIN:
                        pass
                    case _:
                        assert False
            case _:
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        charPos_inStrngWord = CharPos_inStrngWord.MID_BEGIN
                    case CharPos_inStrngWord.MID_BEGIN:
                        pass
                    case CharPos_inStrngWord.BEGIN:
                        pass
                    case CharPos_inStrngWord.BEGIN2END:
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case _:
                        assert False

    assert charPos_inStrngWord == CharPos_inStrngWord.BEGIN
    assert not strngWord2Idx
    for strngWord2Idx in strng2idx:
        assert (len(strngWord2Idx) % 2) == 0

    strng_pretok_split = []
    for strngWord2Idx in strng2idx:
        match len(strngWord2Idx):
            case 2:
                strng_pretok_split.append(strng[strngWord2Idx[0]:
                                          strngWord2Idx[1]+1])
            case 4:
                word = f'{strng[strngWord2Idx[0]: strngWord2Idx[1]+1]}{strng[strngWord2Idx[2]: strngWord2Idx[3]+1]}'
                strng_pretok_split.append(word)
            case _:
                word = ""
                for idx in range(0, len(strngWord2Idx), 2):
                    word = f'{word}{strng[strngWord2Idx[idx]: strngWord2Idx[idx+1]+1]}'
                strng_pretok_split.append(word)
    return strng_pretok_split


def tknLbls2entity_wrds_lbls(
        bch: Dict[str, Any],
        bch_nnOut_tknLbls_ids: torch.Tensor,
        ids2tknLbls: List[str], tokenizer) -> Tuple[List[List[str]], List[
                                                             List[str]]]:
    # purpose of this function is to check that word-labels are generated
    # correctly
    bch_nnOut_entityWrdLbls = []
    bch_userInFiltered_entityWrds = []
    # tokens between two SEP belong to tokens of bch['userIn_pretok']
    nnIn_tknIds_beginEnd_idx = (
            bch['nnIn_ids']['input_ids'] == 102).nonzero()

    for bch_idx in range(bch_nnOut_tknLbls_ids.shape[0]):
        assert bch_nnOut_tknLbls_ids[bch_idx].shape[0] == bch['nnIn_ids'][
                'input_ids'][bch_idx].shape[0]
        prev_wrd_idx = None
        entityWrdLbls = []
        entityWrds = []
        multipleWord_entity = ""

        assert_nnIn_tkns = []
        assert_nnOut_tknLbls = []
        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            # this for-loop is for debugging-only
            assert_nnIn_tkns.append(tokenizer.decode(bch['nnIn_ids'][
                                    'input_ids'][bch_idx, nnIn_tknIds_idx]))
            assert_nnOut_tknLbls.append(ids2tknLbls[bch_nnOut_tknLbls_ids[
                                           bch_idx, nnIn_tknIds_idx].item()])

        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            wrd_idx = bch['mapWords2Tokens'][bch_idx][nnIn_tknIds_idx]
            assert wrd_idx is not None
            if wrd_idx != prev_wrd_idx:  # first token of a word
                nnOut_tknLbl = ids2tknLbls[bch_nnOut_tknLbls_ids[
                                           bch_idx, nnIn_tknIds_idx].item()]
                assert nnOut_tknLbl[0] != "T"
                if nnOut_tknLbl[0] == 'B':
                    if multipleWord_entity:  # previous multipleWord_entity
                        entityWrds.append(multipleWord_entity)
                    entityWrdLbls.append(nnOut_tknLbl)
                    multipleWord_entity = bch[
                                             'userIn_pretok'][bch_idx][wrd_idx]
                elif nnOut_tknLbl[0] == 'I':
                    assert multipleWord_entity
                    multipleWord_entity = (
                            f"{multipleWord_entity} {bch['userIn_pretok'][bch_idx][wrd_idx]}")
                elif nnOut_tknLbl[0] == 'O':  # previous multipleWord_entity
                    if multipleWord_entity:
                        entityWrds.append(multipleWord_entity)
                    multipleWord_entity = ""
                else:
                    strng = (f'invalid token-label {nnOut_tknLbl} starts with '
                             f'{nnOut_tknLbl[0]}')
                    logg.critical(strng)
                    assert False
            else:
                assert (ids2tknLbls[bch_nnOut_tknLbls_ids[
                        bch_idx, nnIn_tknIds_idx].item()])[0] == "T"
            prev_wrd_idx = wrd_idx
        if multipleWord_entity:  # previous multipleWord_entity
            entityWrds.append(multipleWord_entity)
        assert len(entityWrdLbls) == len(entityWrds)
        bch_nnOut_entityWrdLbls.append(entityWrdLbls)
        bch_userInFiltered_entityWrds.append(entityWrds)
    assert len(bch_userInFiltered_entityWrds) == len(bch_nnOut_entityWrdLbls)
    return bch_userInFiltered_entityWrds, bch_nnOut_entityWrdLbls


def CHECK_tokenLabels2wordLabels(
    bch: Dict[str, Any],
    bch_nnOut_tokenLabels_ids: torch.Tensor,
    idx2tokenLabels: List[str]) -> List[List[str]]:
    # purpose of this function is to check that word-labels are generated
    # correctly
    assert bch['nnIn_ids'][
            'input_ids'].shape == bch_nnOut_tokenLabels_ids.shape
    bch_wordLabels = []
    # tokens between two SEP belong to tokens of bch['userIn_pretok']
    userInTokens_beginEnd_idx = (
        bch['nnIn_ids']['input_ids'] == 102).nonzero()

    for bch_idx in range(bch_nnOut_tokenLabels_ids.shape[0]):
        prev_word_idx = None
        wordLabels = []
        for userInToken_idx in range(
                (userInTokens_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   userInTokens_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            word_idx = bch['mapWords2Tokens'][bch_idx][userInToken_idx]
            assert word_idx is not None
            bch_nnOut_tokenLabel = idx2tokenLabels[
                    bch_nnOut_tokenLabels_ids[
                        bch_idx, userInToken_idx].item()]
            if word_idx != prev_word_idx:  # first token of a word
                wordLabels.append(bch_nnOut_tokenLabel)
                # checking for correctness
                # the first token can also be "I"; for example:
                # color = “dark brown”, word-label = B-color, I-color
                remaining_tokenLabel_of_word = (
                        f"I{bch_nnOut_tokenLabel[1:]}" if (
                            bch_nnOut_tokenLabel[0]) != "O" else "O")
            elif word_idx == prev_word_idx:  # not first token of a word
                #*******Since token_labels are compared in Model.predict_step(), is it necessary to do it again here????
                #assert bch_nnOut_tokenLabel == remaining_tokenLabel_of_word
                pass
            else:
                assert False
            prev_word_idx = word_idx
        bch_wordLabels.append(wordLabels)
    assert len(bch['userIn_pretok']) == len(bch_wordLabels)
    for bch_idx in range(len(bch_wordLabels)):
        assert len(bch['userIn_pretok'][bch_idx]) == len(
                bch_wordLabels[bch_idx])
    return (bch_wordLabels)


def userOut_init():
    userOut = {}
    for car_entity_name in syntheticData.groupOf_car_entity_names:
        # dict keys are in same order as syntheticData.groupOf_car_entity_names
        userOut[car_entity_name] = []
    return userOut


def generate_userOut(
        bch_prev_userOut: List[Dict[str, List[str]]],
        bch_userIn_pretok: List[List[str]],
        bch_wordLabels: List[List[str]]) -> List[Dict[str, List[str]]]:
    assert len(bch_wordLabels) == len(bch_userIn_pretok)
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
                           wrdLbl_value if wrdLbl_value else bch_userIn_pretok[
                               bch_idx][wrdLbl_idx])
                case wrdLbl_name if ((
                        wrdLbl_name in syntheticData.
                        groupOf_car_entity_names_with_floatInt_entity_values)
                        or (wrdLbl_name in syntheticData.
                            groupOf_car_entity_names_with_int_entity_values)):
                    assert wrdLbl_value is None
                    bch_userOut[bch_idx][wrdLbl_name].append(bch_userIn_pretok[bch_idx][wrdLbl_idx])

            wrdLbl_idx += 1

    return bch_userOut


def userOut2history(
        bch_userOut: List[Dict[str, List[str]]]) -> List[List[str]]:
    return len(bch_userOut) * [[""]]
