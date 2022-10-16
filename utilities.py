'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import string

logg = getLogger(__name__)

# from a given string: (1) remove all punctuations except keep_puncs, (2) put
# spaces before-and-after add_spaces
replace_punc_with_space, add_spaces_to_punc = {}, {}
keep_puncs, add_spaces = "$%.", "$%"
for punc in string.punctuation:
    if punc not in keep_puncs:
        replace_punc_with_space[punc] = " "
for punc in add_spaces:
    add_spaces_to_punc[punc] = f" {punc} "
table: Dict = str.maketrans(replace_punc_with_space | add_spaces_to_punc)
del replace_punc_with_space, add_spaces_to_punc, keep_puncs, add_spaces, punc


def preTokenize_splitWords(strng: str) -> List[str]:
    # examples: "$400.56." -> ['$', '400.56']   "100%" -> ['100', '%']
    #           "mer#ce.des!" -> ['mer', 'ce.des']   "let's" -> ['let', 's']
    new_strng: List[str] = (strng.translate(table)).split()
    final_strng = []
    for word in new_strng:
        # period and decimal-point have the same punctuation mark; remove
        # periods but not the decimal-points
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
    for token_idx, word_idx in enumerate(map_words2tokens[1: -1]):
        assert word_idx is not None
        if word_idx != prev_word_idx:       # first token of a word
            word_labels.append(token_labels[token_idx])
            prev_word_idx = word_idx
            # checking for correctness
            assert (token_labels[token_idx][0] == "B" or token_labels[token_idx][0] == "O")
            remaining_tokenLabel_of_word = f"I{token_labels[token_idx][1:]}" if token_labels[token_idx][0] == "B" else "O"
        elif word_idx == prev_word_idx:     # not first token of a word
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
