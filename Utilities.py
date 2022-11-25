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


def convert_tokenLabels2wordLabels(
        bch: Dict[str, Any],
        bch_nnOut_tokenLabels_ids: torch.Tensor,
        idx2tokenLabels: List[str]) -> List[List[str]]:
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
            if word_idx != prev_word_idx:  # first token of a word
                wordLabels.append(idx2tokenLabels[
                    bch_nnOut_tokenLabels_ids[
                        bch_idx, userInToken_idx].item()])
            prev_word_idx = word_idx
        bch_wordLabels.append(wordLabels)
    return (bch_wordLabels)


def CHECK_convert_tokenLabels2wordLabels(
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
    assert len(bch_prev_userOut) == len(bch_userIn_pretok)
    # init_userOut = userOut_init(); bch_userOut =
    # [init_userOut for _ in range(len(bch_wordLabels))] Does NOT work
    # because each copy of dict points to same memory location; i.e. writing a
    # value to a key in a dict will write that value to all dicts
    bch_userOut = [userOut_init() for _ in range(len(bch_wordLabels))]

    for bch_idx in range(len(bch_wordLabels)):
        tkn_lbl_idx = 0
        while tkn_lbl_idx < len(bch_wordLabels[bch_idx]):

            def _extract_from_tkn_lbl(label):
                label_prefix = label[0]     # 'B' or 'I'
                if label[-1] == ')':
                    try:
                        label_openParen_idx = label.index('(')
                    except ValueError:
                        logg.critical(
                          '{label} has closing- but not opening-parenthesis')
                        exit()
                    label_name = label[2: label_openParen_idx]
                    label_value = label[label_openParen_idx+1: -1]
                else:
                    label_name = label[2:]
                    label_value = None
                return label_prefix, label_name, label_value
            tkn_lbl_prefix, tkn_lbl_name, tkn_lbl_value = _extract_from_tkn_lbl(
                    bch_wordLabels[bch_idx][tkn_lbl_idx])

            match tkn_lbl_name:
                case tkn_lbl_name if (tkn_lbl_name in syntheticData.groupOf_car_entity_names_with_str_entity_values):
                    # neural-net must memorize value (eg. mercedes) with its
                    # corresponding name (eg. brand)
                    bch_userOut[bch_idx][tkn_lbl_name].append(tkn_lbl_value if tkn_lbl_value  else bch_userIn_pretok[bch_idx][tkn_lbl_idx])
                case tkn_lbl_name if (tkn_lbl_name in syntheticData.groupOf_car_entity_names_with_floatInt_entity_values) or (tkn_lbl_name in syntheticData.groupOf_car_entity_names_with_int_entity_values):
                    assert tkn_lbl_value is None
                    bch_userOut[bch_idx][tkn_lbl_name].append(bch_userIn_pretok[bch_idx][tkn_lbl_idx])

            tkn_lbl_idx += 1

    return bch_userOut


def userOut2history(
        bch_userOut: List[Dict[str, List[str]]]) -> List[List[str]]:
    x = 0
