'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import string
from enum import Enum

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
    can_be_floatPt_num = False

    def resolve_floatPt_num() -> None:
        # ***this function uses strngWord2Idx and strng, and alters strng2idx**
        # ***Assume there can be periods at the beginning and end of word but
        # only one (period or decimal-point) in the middle of word***
        if len(strngWord2Idx) == 2:
            # most likely case
            try:
                float(strng[strngWord2Idx[0]: strngWord2Idx[1]+1])
                strng2idx.append(strngWord2Idx)
                return
            except ValueError:
                pass

        word = ""
        for idx in range(0, len(strngWord2Idx), 2):
            word = f'{word}{strng[strngWord2Idx[idx]: strngWord2Idx[idx+1]+1]}'
        try:
            float(word)     # word with decimal-point or period?
            strng2idx.append(strngWord2Idx)
            return
        except ValueError:
            # To make the implementation easier, assume there is only one
            # period; this word has a period instead of a decimal-point; remove
            # the period and create two words
            assert (strng[strngWord2Idx[0]] != "." and
                    strng[strngWord2Idx[-1]] != ".")
            for i, strng_char in enumerate(strng[strngWord2Idx[
                                                    0]: strngWord2Idx[-1]+1]):
                if strng_char == ".":
                    strng_char_idx = strngWord2Idx[0] + i
                    for j, strng_idx in enumerate(strngWord2Idx):
                        if strng_idx < strng_char_idx:
                            continue
                        elif strng_idx == strng_char_idx:
                            if strngWord2Idx[j] == strngWord2Idx[j+1]:
                                strng2idx.append(strngWord2Idx[: j])
                                if j+2 < len(strngWord2Idx):
                                    strng2idx.append(strngWord2Idx[j+2:])
                            elif len(strngWord2Idx[: j]) % 2:
                                strng2idx.append(strngWord2Idx[: j])
                                temp = [strng_idx+1]
                                temp.extend(strngWord2Idx[j+1:])
                                strng2idx.append(temp)
                            else:
                                temp = strngWord2Idx[: j]
                                temp.append(strng_idx-1)
                                strng2idx.append(temp)
                                strng2idx.append(strngWord2Idx[j+1:])
                        elif strng_idx > strng_char_idx:
                            temp = strngWord2Idx[: j]
                            temp.append(strng_char_idx-1)
                            strng2idx.append(temp)
                            temp = [strng_char_idx+1]
                            temp.extend(strngWord2Idx[j:])
                            strng2idx.append(temp)
                        else:
                            assert False

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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        strngWord2Idx.extend([char_idx, char_idx])
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                    case CharPos_inStrngWord.MID:
                        strngWord2Idx.append(char_idx-1)
                        strng2idx.append(strngWord2Idx)
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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
            case ".":   # period or decimal-point; code is similar to hyphen
                # Problem: is it a period or decimal-point?
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (include) mid;  mid2end (exclude + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (exclude + new word) begin
                match charPos_inStrngWord:
                    case CharPos_inStrngWord.MID:
                        can_be_floatPt_num = True
                    case CharPos_inStrngWord.MID_BEGIN:
                        strngWord2Idx.append(char_idx)
                        charPos_inStrngWord = CharPos_inStrngWord.MID
                        can_be_floatPt_num = True
                    case CharPos_inStrngWord.BEGIN:
                        pass
                    case CharPos_inStrngWord.BEGIN2END:
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case CharPos_inStrngWord.MID2END:
                        strngWord2Idx.append(char_idx-1)
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case _:
                        assert False
            case "-":   # hypen
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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
                            strng2idx.append(strngWord2Idx)
                        charPos_inStrngWord = CharPos_inStrngWord.BEGIN
                        strngWord2Idx = []
                    case CharPos_inStrngWord.MID_BEGIN2END:
                        if can_be_floatPt_num:
                            resolve_floatPt_num()
                            can_be_floatPt_num = False
                        else:
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


def convert_tokenLabels2wordLabels(tokenizer, user_input_pretok: List[str],
                                   token_labels: List[str]) -> List[str]:
    # NOTE: token_labels must NOT have special characters like CLS, SEP, PAD
    map_words2tokens = tokenizer(user_input_pretok,
                                 is_split_into_words=True).word_ids()
    word_labels = []
    prev_word_idx = None
    for token_idx, word_idx in enumerate(map_words2tokens[1:-1]):
        assert word_idx is not None
        if word_idx != prev_word_idx:  # first token of a word
            word_labels.append(token_labels[token_idx])
            prev_word_idx = word_idx
            # checking for correctness
            # the first token can also be "I"; for example:
            # color = “dark brown”, word-label = B-color, I-color
            remaining_tokenLabel_of_word = (
                f"I{token_labels[token_idx][1:]}"
                if token_labels[token_idx][0] != "O" else "O")
        elif word_idx == prev_word_idx:  # not first token of a word
            # checking for correctness
            assert token_labels[token_idx] == remaining_tokenLabel_of_word
        else:
            assert False
    assert len(word_labels) == len(user_input_pretok)
    return (word_labels)


def generate_user_output(user_input_pretok: List[str], word_labels: List[str]):
    assert len(word_labels) == len(user_input_pretok)
    return (" ")


def generate_history(user_output: str) -> str:
    return (" ")


strng = ""
result = preTokenize_splitWords(strng)
my_result = []
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = "   "
result = preTokenize_splitWords(strng)
my_result = []
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = "\n"
result = preTokenize_splitWords(strng)
my_result = []
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = "mercedes,camry red.less than $16000 50% in-cash\n"
result = preTokenize_splitWords(strng)
my_result = ['mercedes', 'camry', 'red', 'less', 'than', '$', '16000', '50', '%', 'in-cash']
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = ",.-$% $44.#6$7$ re%d,g$r@e&^en,$20.4 dark#$-br\t>w?:n -n-o-- "
result = preTokenize_splitWords(strng)
my_result = ['$', '%', '$', '44.67', 'red', 'green', '$', '20.4', 'dark-brwn', 'n-o-']
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = ".44.6.after.the..56.3...222.6. 367.98"
result = preTokenize_splitWords(strng)
my_result = ['44.6', 'after', 'the', '56.3', '222.6', '367.98']
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = "%&()_+%@ \t\n_^"
result = preTokenize_splitWords(strng)
my_result = []
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

x = 1
