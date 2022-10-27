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
        BEGIN = 1           # position of char at beginning of word
        BEGIN2END = 2       # a single char in a word
        MID = 3             # position of char at middle of word
        MID2END = 4         # position of char from middle to end of word
        MID_BEGIN = 5       # position of char at beginning segment of word
        MID_BEGIN2END = 6   # position of char from mid-begin to end of word
    word_pos = Word_pos.BEGIN
    word_idx: List[int] = []
    strng2idx: List[int] = []

    for char_idx, char in enumerate(strng):

        if char == " ":
            assert word_pos == Word_pos.BEGIN
            assert not word_idx
            continue
        if char_idx == (len(strng) - 1) or strng[char_idx+1] == " ":
            match word_pos:
                case Word_pos.BEGIN:
                    word_pos = Word_pos.BEGIN2END
                case Word_pos.MID:
                    word_pos = Word_pos.MID2END
                case Word_pos.MID_BEGIN:
                    word_pos = Word_pos.MID_BEGIN2END
                case _:
                    assert False
        assert (not word_idx) if (word_pos == Word_pos.BEGIN or word_pos == Word_pos.BEGIN2END) else word_idx

        match char:
            case char if char not in string.punctuation and char not in '\t\n':
                # begin (include) mid;  begin2end (include + new word) begin;
                # mid (include) mid;  mid2end (include + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (include + new word) begin
                match word_pos:
                    case Word_pos.MID:
                        pass
                    case Word_pos.BEGIN | Word_pos.MID_BEGIN:
                        word_idx.append(char_idx)
                        word_pos = Word_pos.MID
                    case Word_pos.MID2END:
                        word_idx.append(char_idx)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID_BEGIN2END:
                        word_idx.extend([char_idx, char_idx])
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False
            case ",":   # comma
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude + new word) begin;
                # mid2end (exclude + new word) begin;
                # mid_begin (exclude + new word) begin;
                # mid_begin2end (exclude + new word) begin;
                match word_pos:
                    case Word_pos.MID2END | Word_pos.MID:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                    case Word_pos.MID_BEGIN | Word_pos.MID_BEGIN2END:
                        strng2idx.append(word_idx)
                    case Word_pos.BEGIN | Word_pos.BEGIN2END:
                        pass
                    case _:
                        assert False
                word_pos = Word_pos.BEGIN
                word_idx = []
            case ".":   # period
                pass
            case "-":   # hypen
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (include) mid;  mid2end (exclude + new word) begin;
                # mid_begin (include) mid;
                # mid_begin2end (exclude + new word) begin
                match word_pos:
                    case Word_pos.MID:
                        pass
                    case Word_pos.MID_BEGIN:
                        word_idx.append(char_idx)
                        word_pos = Word_pos.MID
                    case Word_pos.BEGIN:
                        pass
                    case Word_pos.BEGIN2END:
                        word_pos = Word_pos.BEGIN
                    case Word_pos.MID2END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID_BEGIN2END:
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case _:
                        assert False
            case "$":   # dollar
                # begin (include + new word) begin;
                # begin2end (include + new word) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match word_pos:
                    case Word_pos.BEGIN:
                        strng2idx.append([char_idx, char_idx])
                    case Word_pos.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                    case Word_pos.MID2END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID_BEGIN2END:
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
                # begin (exclude) begin; begin2end (include + new word) begin;
                # mid (exclude) mid_begin; mid2end (include + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (include + two words) begin
                match word_pos:
                    case Word_pos.MID2END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.BEGIN2END:
                        strng2idx.append([char_idx, char_idx])
                        word_pos = Word_pos.BEGIN
                    case Word_pos.MID_BEGIN2END:
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
                    case _:
                        assert False
            case _:
                # begin (exclude) begin;  begin2end (exclude) begin;
                # mid (exclude) mid_begin; mid2end (exclude + new word) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (exclude + new word) begin
                match word_pos:
                    case Word_pos.MID2END:
                        word_idx.append(char_idx-1)
                        strng2idx.append(word_idx)
                        word_pos = Word_pos.BEGIN
                        word_idx = []
                    case Word_pos.MID_BEGIN2END:
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
                    case Word_pos.BEGIN2END:
                        word_pos = Word_pos.BEGIN
                    case _:
                        assert False

    assert word_pos == Word_pos.BEGIN
    assert not word_idx
    for word_idx in strng2idx:
        assert (len(word_idx) % 2) == 0

    strng_pretok_split = []
    for word_idx in strng2idx:
        match len(word_idx):
            case 2:
                strng_pretok_split.append(strng[word_idx[0]: word_idx[1]+1])
            case 4:
                word = f'{strng[word_idx[0]: word_idx[1]+1]}{strng[word_idx[2]: word_idx[3]+1]}'
                strng_pretok_split.append(word)
            case _:
                word = ""
                for idx in range(0, len(word_idx), 2):
                    word = f'{word}{strng[word_idx[idx]: word_idx[idx+1]+1]}'
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


strng = "mercedes,camry red.less than $16000 50% in-cash\n"
result = preTokenize_splitWords(strng)
my_result = ['mercedes', 'camry', 'red.less', 'than', '$', '16000', '50', '%', 'in-cash']
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

strng = ",.-$% $44.#6$7$ re%d,g$r@e&^en,$20.4 dark#$-br\t>w?:n -n-o-- "
result = preTokenize_splitWords(strng)
my_result = ['$', '%', '$', '44.67', 'red', 'green', '$', '20.4', 'dark-brwn', 'n-o-']
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

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

strng = "%&()_+%@ \t\n_^"
result = preTokenize_splitWords(strng)
my_result = []
print(f'{strng}\n{my_result}\n{result}\n{result == my_result}\n')

x = 1
