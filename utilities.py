'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import string
from enum import Enum

logg = getLogger(__name__)


def preTokenize_splitWords(strng: str) -> List[str]:
    class Word_pos(Enum):
        BEGIN = 1       # position of char at beginning of word
        BEGIN_END = 2   # a single char in a word
        MID = 3         # position of char at middle of word
        MID_BEGIN = 4   # position of char at beginning segment of word
        END = 5         # position of char at end of word
    word_pos = Word_pos.BEGIN
    word_idx: List[int] = []
    strng2idx: List[int] = []

    for char_idx in range(len(strng)):

        if strng[char_idx] == " ":
            assert word_pos == Word_pos.BEGIN
            assert not word_idx
            continue
        if char_idx == (len(strng) - 1) or strng[char_idx+1] == " ":
            if word_pos == Word_pos.BEGIN:
                word_pos = Word_pos.BEGIN_END
            else:
                word_pos = Word_pos.END
        assert (not word_idx) if (word_pos == Word_pos.BEGIN or word_pos == Word_pos.BEGIN_END) else word_idx

        match strng[char_idx]:
            case char if char not in string.punctuation:
                # begin->mid: include, begin_end->begin: include + new word,
                # mid->mid: include, mid_begin->mid: include,
                # end->begin: include + new word
                match word_pos:
                    case Word_pos.MID:
                        pass
                    case Word_pos.BEGIN | Word_pos.MID_BEGIN:
                        word_idx.append(char_idx)
                        word_pos = Word_pos.MID
                    case Word_pos.END:
                        # MID->END or MID_BEGIN->END
                        if len(word_idx) % 2:
                            word_idx.append(char_idx)
                        else:
                            word_idx.append([char_idx, char_idx])
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.BEGIN_END:
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False
            case ",":   # comma
                # begin->begin: exclude, begin_end->begin: exclude,
                # mid->begin: exclude + new word,
                # mid_begin->begin: exclude + new word,
                # end->begin: exclude + new word
                match word_pos:
                    case Word_pos.END | Word_pos.MID:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                    case Word_pos.MID_BEGIN:
                        strng2idx.append(word_idx)
                    case Word_pos.BEGIN:
                        pass
                    case Word_pos.BEGIN_END:
                        pass
                    case _:
                        assert False
                word_pos = Word_pos.BEGIN
                word_idx = []
            case ".":   # period
                pass
            case "-":   # hypen
                # begin->begin: exclude, begin_end->begin: exclude,
                # mid->mid: include, mid_begin->mid: include,
                # end->begin: exclude + new word
                match word_pos:
                    case Word_pos.MID:
                        pass
                    case Word_pos.MID_BEGIN:
                        word_idx.append(char_idx)
                        word_pos = Word_pos.MID
                    case Word_pos.BEGIN:
                        pass
                    case Word_pos.BEGIN_END:
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False
            case "$":   # dollar
                # begin->begin: include + new word,
                # begin_end->begin: include + new word,
                # mid->mid_begin: exclude, mid_begin->mid_begin: exclude,
                # end->begin: exclude + new word
                match word_pos:
                    case Word_pos.BEGIN:
                        strng2idx.append([char_idx, char_idx])
                    case Word_pos.BEGIN_END:
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID:
                        word_idx.append(char_idx-1)
                        word_pos = Word_pos.MID_BEGIN
                    case Word_pos.MID_BEGIN:
                        pass
                    case _:
                        assert False
            case "%":   # percent
                # begin->begin: exclude, begin_end->begin: exclude,
                # mid->mid_begin: exclude, mid_begin->mid_begin: exclude,
                # end->begin: include + two words
                match word_pos:
                    case Word_pos.END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID:
                        word_idx.append(char_idx-1)
                        word_pos = Word_pos.MID_BEGIN
                    case Word_pos.MID_BEGIN:
                        pass
                    case Word_pos.BEGIN:
                        pass
                    case Word_pos.BEGIN_END:
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False
            case _:
                # begin->begin: exclude, begin_end->begin: exclude,
                # mid->mid_begin: exclude, mid_begin->mid_begin: exclude,
                # end->begin: exclude + new words
                match word_pos:
                    case Word_pos.END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID:
                        word_idx.append(char_idx-1)
                        word_pos = Word_pos.MID_BEGIN
                    case Word_pos.MID_BEGIN:
                        pass
                    case Word_pos.BEGIN:
                        pass
                    case Word_pos.BEGIN_END:
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False

    assert word_pos == Word_pos.BEGIN
    assert not word_idx
    assert (len(strng2idx) % 2) == 0

    strng_pretok_split = []
    for word_idx in strng2idx:
        match len(word_idx):
            case 2:
                #???? What if word_idx[1] is at end of string; then word_idx[1]+1 gives index error
                strng_pretok_split.append(strng[word_idx[0]: word_idx[1]+1])
            case 4:
                strng_pretok_split.append(strng[word_idx[0]: word_idx[1]+1])
                strng_pretok_split.append(strng[word_idx[2]: word_idx[3]+1])
            case _:
                assert (len(word_idx) % 2) == 0
                word = ""
                for idx in range(0, len(word_idx), 2):
                    word = f'{word}{strng[word_idx[idx]: word_idx[idx]+1]}'
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


strng = ",.-$% $44.#6$7$ re%d,g$een,$20.4 dark#$-br\t>w?:n -n-o--"
print(preTokenize_splitWords(strng))
print(preTokenize_splitWords(""))
print(preTokenize_splitWords(","))
