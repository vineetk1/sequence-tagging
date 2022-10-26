'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import string
from enum import Enum

logg = getLogger(__name__)

used_punctuations = ".,-$%"
unused_punctuations = ""
for punc in string.punctuation:
    if punc not in used_punctuations:
        unused_punctuations = f'{unused_punctuations}{punc}'
del used_punctuations


def preTokenize_splitWords(strng: str) -> List[str]:
    class Word_pos(Enum):
        BEGIN = 1       # position of char at beginning of word
        MID = 2         # position of char at middle of word
        MID_BEGIN = 3   # position of char at beginning segment of word
    word_pos = Word_pos.BEGIN
    str2out_idx: List[int] = []
    word_idx: List[int] = []
    idx_of_last_char = len(strng) - 1

    for char_idx in range(len(strng)):
        assert (not word_idx) if word_pos == Word_pos.BEGIN else word_idx
        match strng[char_idx]:
            case ",":   # comma
                # begin: exclude, mid => end: exclude + new word
                match word_pos:
                    case Word_pos.MID:
                        str2out_idx.append(word_idx.append(char_idx-1))
                    case Word_pos.MID_BEGIN:
                        str2out_idx.append(word_idx)
                    case Word_pos.BEGIN:
                        assert not word_idx
                    case _:
                        assert False
                word_pos = Word_pos.BEGIN
                word_idx = []
            case ".":   # period
                pass
            case "-":   # hypen
                # begin: exclude, mid: include, end: exclude
                match word_pos:
                    case Word_pos.MID:
                        if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                            str2out_idx.append(word_idx.append(char_idx-1))
                            word_pos = Word_pos.BEGIN
                            word_idx = []
                    case Word_pos.MID_BEGIN:
                        if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                            str2out_idx.append(word_idx)
                            word_pos = Word_pos.BEGIN
                            word_idx = []
                        else:
                            word_idx.append(char_idx)
                            word_pos = Word_pos.MID
                    case Word_pos.BEGIN:
                        assert not word_idx
                    case _:
                        assert False
            case "$":   # dollar
                # begin: new word, mid: exclude, end: exclude
                match word_pos:
                    case Word_pos.BEGIN:
                        str2out_idx.append([char_idx, char_idx])
                    case Word_pos.MID:
                        word_idx.append(char_idx-1)
                        word_pos = Word_pos.MID_BEGIN
                    case Word_pos.MID_BEGIN:
                        pass
                    case _:
                        assert False
            case "%":   # percent
                # begin: exclude, mid: exclude, end: include + two words
                match word_pos:
                    case Word_pos.MID:
                        if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                            str2out_idx.append(word_idx.append(char_idx-1))
                            str2out_idx.append([char_idx, char_idx])
                            word_pos = Word_pos.BEGIN
                            word_idx = []
                        else:
                            word_idx.append(char_idx-1)
                            word_pos = Word_pos.MID_BEGIN
                    case Word_pos.MID_BEGIN:
                        if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                            str2out_idx.append(word_idx)
                            str2out_idx.append([char_idx, char_idx])
                            word_pos = Word_pos.BEGIN
                            word_idx = []
                    case Word_pos.BEGIN:
                        if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                            str2out_idx.append([char_idx, char_idx])
                        else:
                            assert not word_idx
                    case _:
                        assert False
            case " ":   # space
                match word_pos:
                    case Word_pos.MID:
                        assert word_idx
                        str2out_idx.append(word_idx.append(char_idx-1))
                    case Word_pos.BEGIN:
                        assert not word_idx
                    case Word_pos.MID_BEGIN:
                        assert word_idx
                        str2out_idx.append(word_idx)
                    case _:
                        assert False
                word_pos = Word_pos.BEGIN
                word_idx = []
            case _:
                if strng[char_idx] not in unused_punctuations:
                    # begin: include, mid: include, end: include
                    match word_pos:
                        case Word_pos.MID:
                            pass
                        case Word_pos.BEGIN | Word_pos.MID_BEGIN:
                            assert (not word_idx) if word_pos == Word_pos.BEGIN else word_idx
                            word_idx.append(char_idx)
                            word_pos = Word_pos.MID
                        case _:
                            assert False
                else:
                    # begin: exclude, mid: exclude, end: exclude + new word
                    match word_pos:
                        case Word_pos.MID:
                            if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                                str2out_idx.append(word_idx.append(char_idx-1))
                                word_pos = Word_pos.BEGIN
                                word_idx = []
                            else:
                                word_idx.append(char_idx-1)
                                word_pos = Word_pos.MID_BEGIN
                        case Word_pos.MID_BEGIN:
                            if char_idx == idx_of_last_char or strng[char_idx+1] == " ":
                                str2out_idx.append(word_idx)
                                word_pos = Word_pos.BEGIN
                                word_idx = []
                        case Word_pos.BEGIN:
                            pass
                        case _:
                            assert False

    match word_pos:
        case Word_pos.BEGIN | Word_pos.MID_BEGIN:
            assert not word_idx
        case Word_pos.MID:
            str2out_idx.append(word_idx.append(char_idx-1))
        case _:
            assert False

    assert (len(str2out_idx) % 2) == 0
    strng_pretok_split = []
    for word_idxs in str2out_idx:
        if len(word_idxs) == 2:
            strng_pretok_split.append(strng[word_idxs[0]: word_idxs[1]+1])
        else:
            word = ""
            for word_idx in range(0, len(word_idxs), 2):
                word = f'{word}{strng[word_idxs[word_idx]: word_idxs[word_idx]+1]}'
            strng_pretok_split.append(word)
    return strng_pretok_split





def xpreTokenize_splitWords(strng: str) -> List[str]:
    # *****NOTE**** The best way to implement is to pick one character at a time and decide what to do; also if a character is at the end of a word then it could be treated differently than when it is in the middle of the word; for example: user types "red,blue"then the comma should be replaced with space

    # examples: "$400.56." -> ['$', '400.56']   "100%" -> ['100', '%']
    #           "mer#ce.des!" -> ['mer', 'ce.des']   "let's" -> ['let', 's']
    new_strng: List[str] = (strng.translate(table)).split()
    final_strng = []
    for word in new_strng:
        # period and decimal-point have the same punctuation mark; remove
        # periods from words that end in periods, which will not include
        # decimal-points because they are not at the end of words
        if word.endswith("."):
            final_strng.append(word.strip("."))
        else:
            final_strng.append(word)
    return final_strng


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

strng = ",.-$% $44.#6$7$ re%d,g$een,$20.4 dark#$-br\o>w?:n -n-o--"
print(preTokenize_splitWords(strng))
print(preTokenize_splitWords(""))
print(preTokenize_splitWords(","))
