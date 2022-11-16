'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import string
from enum import Enum
import torch

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
                # mid (exclude + new word) begin;
                # mid2end (exclude + new word) begin;
                # mid_begin (exclude + new word) begin;
                # mid_begin2end (exclude + new word) begin;
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strng2idx.append(strngWord2Idx)
                    case CharPos_inStrngWord.MID_BEGIN:
                        strng2idx.append(strngWord2Idx)
                    case (CharPos_inStrngWord.BEGIN |
                          CharPos_inStrngWord.BEGIN2END):
                        pass
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
        tokenizer, batch: Dict[str, Any],
        batch_nnOut_tokenLabels_ids: torch.Tensor,
        all_token_labels: List[str]) -> List[List[str]]:
    batch_word_labels = []
    # tokens between two SEP belong to tokens of batch['user_input_pretok']
    userInputTokens_beginEnd_idx = (
            batch['nnIn_ids']['input_ids'] == 102).nonzero()

    for bch_idx in range(batch_nnOut_tokenLabels_ids.shape[0]):
        prev_word_idx = None
        word_labels = []
        for userInputToken_idx in range(
                (userInputTokens_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   userInputTokens_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            word_idx = batch['mapWords2Tokens'][bch_idx][userInputToken_idx]
            if word_idx != prev_word_idx:  # first token of a word
                word_labels.append(all_token_labels[
                    batch_nnOut_tokenLabels_ids[
                        bch_idx, userInputToken_idx].item()])
            prev_word_idx = word_idx
        batch_word_labels.append(word_labels)
    return (batch_word_labels)


def CHECK_convert_tokenLabels2wordLabels(
        tokenizer, batch: Dict[str, Any],
        batch_nnOut_tokenLabels_ids: torch.Tensor,
        all_token_labels: List[str]) -> List[List[str]]:
    # purpose of this function is to check that word-labels are generated
    # correctly
    assert batch['nnIn_ids'][
            'input_ids'].shape == batch_nnOut_tokenLabels_ids.shape
    batch_word_labels = []
    # tokens between two SEP belong to tokens of batch['user_input_pretok']
    userInputTokens_beginEnd_idx = (
            batch['nnIn_ids']['input_ids'] == 102).nonzero()

    for bch_idx in range(batch_nnOut_tokenLabels_ids.shape[0]):
        prev_word_idx = None
        word_labels = []
        for userInputToken_idx in range(
                (userInputTokens_beginEnd_idx[bch_idx * 2, 1] + 1).item(), (
                   userInputTokens_beginEnd_idx[(bch_idx * 2) + 1, 1]).item()):
            word_idx = batch['mapWords2Tokens'][bch_idx][userInputToken_idx]
            assert word_idx is not None
            batch_nnOut_tokenLabel = all_token_labels[
                    batch_nnOut_tokenLabels_ids[
                        bch_idx, userInputToken_idx].item()]
            if word_idx != prev_word_idx:  # first token of a word
                word_labels.append(batch_nnOut_tokenLabel)
                # checking for correctness
                # the first token can also be "I"; for example:
                # color = “dark brown”, word-label = B-color, I-color
                remaining_tokenLabel_of_word = (
                        f"I{batch_nnOut_tokenLabel[1:]}" if (
                            batch_nnOut_tokenLabel[0]) != "O" else "O")
            elif word_idx == prev_word_idx:  # not first token of a word
                #*******Since token_labels are compared in Model.predict_step(), is it necessary to do it again here????
                #assert batch_nnOut_tokenLabel == remaining_tokenLabel_of_word
                pass
            else:
                assert False
            prev_word_idx = word_idx
        batch_word_labels.append(word_labels)
    assert len(batch['user_input_pretok']) == len(batch_word_labels)
    for batch_idx in range(len(batch_word_labels)):
        assert len(batch['user_input_pretok'][batch_idx]) == len(
                batch_word_labels[batch_idx])
    return (batch_word_labels)


def generate_userOutputs_histories(user_input_pretok: List[str], word_labels:
                                   List[str]) -> Tuple[str, List[str]]:
    assert len(word_labels) == len(user_input_pretok)
    user_output = " "
    history = [" "]
    return user_output, history
