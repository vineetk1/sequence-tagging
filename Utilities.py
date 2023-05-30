'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Union
import string
from enum import Enum
import torch
import copy
import generate_dataset.Synthetic_dataset as syntheticData

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
                # mid (exclude) mid_begin;
                # mid2end (include + 2 new words) begin;
                # mid_begin (exclude) mid_begin;
                # mid_begin2end (include + 2 two words) begin
                match charPos_inUserInWrd:
                    case CharPos_inUserInWrd.BEGIN:
                        userIn2idx.append([char_idx, char_idx])
                    case CharPos_inUserInWrd.BEGIN2END:
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                    case CharPos_inUserInWrd.MID2END:
                        userIn_wrd2Idx.append(char_idx-1)
                        userIn2idx.append(userIn_wrd2Idx)
                        userIn2idx.append([char_idx, char_idx])
                        charPos_inUserInWrd = CharPos_inUserInWrd.BEGIN
                        userIn_wrd2Idx = []
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
                            if userIn_filter_split[
                               idx_in_userIn2idx-1] in syntheticData.all_units:
                                if idx_in_userIn2idx-1:
                                    float(userIn_filter_split[
                                                        idx_in_userIn2idx-2])
                                else:
                                    remove_word_idxs.append(idx_in_userIn2idx)
                                    continue
                            else:
                                float(userIn_filter_split[idx_in_userIn2idx-1])
                        else:
                            remove_word_idxs.append(idx_in_userIn2idx)
                            continue
                        if idx_in_userIn2idx < len(userIn_filter_split) - 1:
                            if userIn_filter_split[
                               idx_in_userIn2idx+1] in syntheticData.all_units:
                                if idx_in_userIn2idx < len(
                                                    userIn_filter_split) - 2:
                                    float(
                                      userIn_filter_split[idx_in_userIn2idx+2])
                                else:
                                    remove_word_idxs.append(idx_in_userIn2idx)
                                    continue
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
        bch_userIn_filtered_wrds: List[List[str]],
        bch_nnOut_tknLblIds: torch.Tensor,
        tknLblId2tknLbl: List[str],
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
            D_tknLbl_True = tknLblId2tknLbl[DEBUG_bch_tknLblIds_True[bch_idx, D_nnIn_tknIds_idx]]
            D_nnOut_tknLbl = tknLblId2tknLbl[bch_nnOut_tknLblIds[bch_idx, D_nnIn_tknIds_idx]]
            D_userIn_filtered_wrd = bch_userIn_filtered_wrds[bch_idx][bch_map_tknIdx2wrdIdx[bch_idx][D_nnIn_tknIds_idx]]
            D_bch_associate[-1].append((D_nnIn_tknIds_idx, D_userIn_filtered_wrd, D_nnIn_tkn, D_tknLbl_True, D_nnOut_tknLbl))
    # ******************remove DEBUG code ending  here**********************
    # NOTE: Remove all ASSERTS from Production code of this function
    #   Right now one assert is commented at: assert not nnOut_tknLbl == 'T'
    assert bch_nnOut_tknLblIds.shape == bch_nnIn_tknIds.shape

    bch_nnOut_entityLbls: List[List[str]] = []
    bch_nnOut_userIn_filtered_entityWrds: List[List[str]] = []
    entityWrd: str = None
    entityLbl: str = None
    userIn_filtered_idx: int = None
    max_count_wrongPredictions_plus1: int = 2 + 1

    # tknIds between two SEP belong to tknIds of words in
    # bch['userIn_filtered_wrds']
    nnIn_tknIds_idx_beginEnd: torch.Tensor = (bch_nnIn_tknIds == 102).nonzero()
    tknLblId_of_O: int = tknLblId2tknLbl.index("O")

    for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
        multipleWord_entity: str = ""
        prev_userIn_filtered_idx: int = None
        prev_BIO: str = None
        bch_nnOut_entityLbls.append([])
        bch_nnOut_userIn_filtered_entityWrds.append([])
        count_wrongPredictions: int = 0

        for nnIn_tknIds_idx in range(
                (nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1), (
                   nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
            if (userIn_filtered_idx := bch_map_tknIdx2wrdIdx[bch_idx][
                 nnIn_tknIds_idx]) == prev_userIn_filtered_idx:
                continue    # ignore tknId that is not first-token-of-the-word
            prev_userIn_filtered_idx = userIn_filtered_idx
            if bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx] == tknLblId_of_O:
                if multipleWord_entity:  # previous multipleWord_entity
                    bch_nnOut_userIn_filtered_entityWrds[-1].append(
                                                        multipleWord_entity)
                    multipleWord_entity = ""
                prev_BIO = "O"   # next tkn is "B" or "O"
                continue    # ignore tknLblId of "O"

            nnOut_tknLbl = tknLblId2tknLbl[bch_nnOut_tknLblIds[
                                                     bch_idx, nnIn_tknIds_idx]]
            if nnOut_tknLbl[-1] == ')':
                # there MUST be an opening-parenthesis, else error
                tknLbl_openParen_idx = nnOut_tknLbl.index('(')
                entityWrd = nnOut_tknLbl[tknLbl_openParen_idx+1: -1]
                entityLbl = nnOut_tknLbl[2: tknLbl_openParen_idx]
            else:
                entityWrd = bch_userIn_filtered_wrds[
                                                  bch_idx][userIn_filtered_idx]
                entityLbl = nnOut_tknLbl[2:]
            if nnOut_tknLbl[0] == 'B':
                if multipleWord_entity:  # previous multipleWord_entity
                    bch_nnOut_userIn_filtered_entityWrds[-1].append(
                                                       multipleWord_entity)
                bch_nnOut_entityLbls[-1].append(entityLbl)
                multipleWord_entity = entityWrd
                prev_BIO = "B"    # next tkn is "B" or "I" or "O"
            elif nnOut_tknLbl[0] == 'I':
                if prev_BIO == "B" or prev_BIO == "I":
                    assert multipleWord_entity
                    if entityLbl != bch_nnOut_entityLbls[-1][-1]:
                        count_wrongPredictions = (
                                max_count_wrongPredictions_plus1)
                        break
                    else:
                        multipleWord_entity = (
                                f"{multipleWord_entity} {entityWrd}")
                    prev_BIO = "I"    # next tkn is "B" or "I" or "O"
                elif prev_BIO == "O":
                    count_wrongPredictions = max_count_wrongPredictions_plus1
                    break
                else:   # prev_BIO == None
                    # expected "B" or "O" at start-of-sentence but model
                    # predicts "I"
                    #assert prev_BIO is None
                    count_wrongPredictions = max_count_wrongPredictions_plus1
                    break
            else:
                #assert False
                count_wrongPredictions = max_count_wrongPredictions_plus1
                break
        if multipleWord_entity:  # previous multipleWord_entity
            bch_nnOut_userIn_filtered_entityWrds[-1].append(
                    multipleWord_entity)
        if count_wrongPredictions >= max_count_wrongPredictions_plus1:
            del bch_nnOut_userIn_filtered_entityWrds[-1]
            bch_nnOut_userIn_filtered_entityWrds.append(None)
            del bch_nnOut_entityLbls[-1]
            bch_nnOut_entityLbls.append(None)
            assert len(bch_nnOut_userIn_filtered_entityWrds) == len(
                                                    bch_nnOut_entityLbls)
        else:
            assert len(bch_nnOut_userIn_filtered_entityWrds[bch_idx]) == len(
                                              bch_nnOut_entityLbls[bch_idx])
    assert len(bch_nnOut_userIn_filtered_entityWrds) == len(
                                                       bch_nnOut_entityLbls)
    return bch_nnOut_userIn_filtered_entityWrds, bch_nnOut_entityLbls


def userOut_init():
    userOut = {}
    for carEntityLbl in syntheticData.carEntityLbls:
        userOut[carEntityLbl] = []
    return userOut


def generate_userOut(
        bch_prevTrnUserOut: List[Dict[str, List[str]]],
        bch_nnOut_userIn_filtered_entityWrds: List[List[str]],
        bch_nnOut_entityLbls: List[List[str]]) -> List[Dict[str, List[str]]]:
    bch_nnOut_userOut = copy.deepcopy(bch_prevTrnUserOut)
    assert len(bch_nnOut_entityLbls) == len(
                                          bch_nnOut_userIn_filtered_entityWrds)
    assert len(bch_nnOut_entityLbls) == len(bch_nnOut_userOut)
    # **** For Deployment: (1) Code is written so Asserts, along with
    # associated If statements, can be removed; (2) the calling function
    # "transition" has a dict parameter with a number of variables; this dict
    # is for debugging purposes only

    for bch_idx in range(len(bch_nnOut_entityLbls)):
        if bch_nnOut_entityLbls[bch_idx] is None:
            # ******* do not change bch_nnOut_userOut[bch_idx] so the user does
            # not see any change on his end; however tell the user that the
            # model cannot understand his text??? *****
            continue
        assert len(bch_nnOut_entityLbls[bch_idx]) == len(
                              bch_nnOut_userIn_filtered_entityWrds[bch_idx])
        wrdLbl_idx = 0
        cmd: str = None
        unit: str = None
        carEntityNums: List[str] = []
        carEntityNumsLbl: str = None
        carEntityNumsNeeded: int = None
        while wrdLbl_idx < len(bch_nnOut_entityLbls[bch_idx]):
            entityWrd = bch_nnOut_userIn_filtered_entityWrds[
                                                        bch_idx][wrdLbl_idx]
            match entityLbl := bch_nnOut_entityLbls[bch_idx][wrdLbl_idx]:

                case entityLbl if (
                        entityLbl in syntheticData.
                        carEntityLbls_for_nonNumEntityWrds):
                    if carEntityNumsLbl:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                    cmd, carEntityNumsNeeded, unit = None, None, None
                    carEntityNums.clear()
                    carEntityNumsLbl = None
                    if entityWrd not in bch_nnOut_userOut[bch_idx][entityLbl]:
                        bch_nnOut_userOut[bch_idx][entityLbl].append(entityWrd)

                case entityLbl if ((
                        entityLbl in syntheticData.
                        carEntityLbls_for_numEntityWrds)):
                    if ((carEntityNumsLbl and carEntityNumsLbl != entityLbl) or
                            (carEntityNumsNeeded and
                                len(carEntityNums) == carEntityNumsNeeded)):
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                    assert (carEntityNumsNeeded and len(carEntityNums) <
                            carEntityNumsNeeded) or (not carEntityNumsNeeded)
                    carEntityNums.append(entityWrd)
                    carEntityNumsLbl = entityLbl
                    if (carEntityNumsNeeded and len(carEntityNums) == carEntityNumsNeeded) and ((unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit)) or (carEntityNumsLbl not in syntheticData.carEntityLbls_require_unit)):
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None

                case 'units_price' | 'units_mileage':
                    if (unit and (unit not in syntheticData.nonCarEntityLbls_mapTo_entityWrds[entityLbl])) or (carEntityNumsLbl and (carEntityNumsLbl not in syntheticData.carEntityLbls_require_unit)):
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None
                    unit = entityWrd
                    if carEntityNumsNeeded and len(
                            carEntityNums) == carEntityNumsNeeded:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None

                case 'more' | 'less':
                    if carEntityNumsNeeded is not None:
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None
                    elif len(carEntityNums) > 1:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums[:-1], carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded = None, None
                        carEntityNums = carEntityNums[-1:]
                    else:
                        pass

                    if not carEntityNums:
                        cmd, carEntityNumsNeeded, unit = entityLbl, 1, None
                        carEntityNumsLbl = None
                    else:   # carEntityNums
                        idx = wrdLbl_idx + 1
                        seg_ends_with_cmd = None
                        found_numLbl, found_unitCatLbl = False, False
                        while (idx < len(bch_nnOut_entityLbls[bch_idx])) and (
                                seg_ends_with_cmd is None):
                            lbl = bch_nnOut_entityLbls[bch_idx][idx]
                            if lbl in (syntheticData.
                                       carEntityLbls_for_numEntityWrds):
                                if found_numLbl:
                                    seg_ends_with_cmd = True
                                else:
                                    found_numLbl = True
                            elif lbl in syntheticData.unit_categories:
                                if found_unitCatLbl:
                                    seg_ends_with_cmd = True
                                else:
                                    found_unitCatLbl = True
                            elif lbl == 'more' or lbl == 'less':
                                seg_ends_with_cmd = True
                            else:
                                if (found_numLbl and lbl not in
                                        syntheticData.cmds_after_carEntityNum):
                                    seg_ends_with_cmd = False
                                else:
                                    seg_ends_with_cmd = True
                            idx += 1
                        if seg_ends_with_cmd or (seg_ends_with_cmd is None and not found_numLbl):
                            transition(bch_nnOut_userOut[bch_idx], entityLbl,
                                       unit, carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                            cmd, carEntityNumsNeeded, unit = None, None, None
                            carEntityNums.clear()
                            carEntityNumsLbl = None
                        elif not seg_ends_with_cmd or (seg_ends_with_cmd is None and found_numLbl):
                            transition(bch_nnOut_userOut[bch_idx], None, unit,
                                       carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                            cmd, carEntityNumsNeeded, unit = entityLbl, 1, None
                            carEntityNums.clear()
                            carEntityNumsLbl = None
                        else:
                            assert False

                case 'range1':
                    if carEntityNumsLbl:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                    cmd, carEntityNumsNeeded, unit = entityLbl, 2, None
                    carEntityNums.clear()
                    carEntityNumsLbl = None

                case 'range2':
                    if entityWrd == "to" and cmd == 'range1' and len(carEntityNums) == 1:
                        # special case
                        wrdLbl_idx += 1
                        continue

                    if (carEntityNumsNeeded is not None) or not carEntityNums:
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None
                    elif len(carEntityNums) == 1:
                        cmd, carEntityNumsNeeded = entityLbl, 2
                    elif len(carEntityNums) > 1:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums[:-1], carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded = entityLbl, 2
                        carEntityNums = carEntityNums[-1:]
                    else:
                        pass

                case 'remove':
                    if carEntityNumsLbl:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None

                    wrdLbl_idx += 1
                    if wrdLbl_idx >= len(bch_nnOut_entityLbls[bch_idx]):
                        break
                    entityWrd = bch_nnOut_userIn_filtered_entityWrds[
                                                        bch_idx][wrdLbl_idx]
                    match entityLbl := bch_nnOut_entityLbls[
                                                        bch_idx][wrdLbl_idx]:
                        case 'everything':
                            # remove everything
                            for k in bch_nnOut_userOut[bch_idx].keys():
                                bch_nnOut_userOut[bch_idx][k].clear()
                        case 'carEntityLbl':
                            # remove brand                 delete brands
                            assert (entityWrd in syntheticData.
                                    nonCarEntityLbls_mapTo_entityWrds[
                                        entityLbl])
                            if entityWrd[-1] == "s":
                                bch_nnOut_userOut[
                                            bch_idx][entityWrd[:-1]].clear()
                            else:
                                bch_nnOut_userOut[bch_idx][entityWrd].clear()
                        case _:
                            #assert False
                            wrdLbl_idx -= 1

                case "restore":
                    if carEntityNumsLbl:
                        transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                                   carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
                        cmd, carEntityNumsNeeded, unit = None, None, None
                        carEntityNums.clear()
                        carEntityNumsLbl = None

                    wrdLbl_idx += 1
                    if wrdLbl_idx < len(bch_nnOut_entityLbls[bch_idx]) and (
                       bch_nnOut_entityLbls[bch_idx][wrdLbl_idx] == 'setting'):
                        try:
                            num = int(bch_nnOut_userIn_filtered_entityWrds[
                                bch_idx][wrdLbl_idx])
                            # following is dummy userOut
                            bch_nnOut_userOut[bch_idx] = {'brand': ['bentley'], 'model': ['nv3500 hd passenger'], 'color': ['tuxedo black metallic'], 'style': ['vanminivan'], 'mileage': ['less 26632.01 kilometers'], 'price': ['more 5000 $'], 'year': ['2015-2023']}
                        except ValueError:
                            wrdLbl_idx -= 1
                    else:
                        wrdLbl_idx -= 1

                case _:
                    #assert False
                    pass

            wrdLbl_idx += 1

        if carEntityNumsLbl:
            transition(bch_nnOut_userOut[bch_idx], cmd, unit,
                       carEntityNums, carEntityNumsLbl, {"bch_nnOut_userIn_filtered_entityWrds[bch_idx]": bch_nnOut_userIn_filtered_entityWrds[bch_idx], "bch_nnOut_entityLbls[bch_idx]": bch_nnOut_entityLbls[bch_idx], "wrdLbl_idx": wrdLbl_idx, "carEntityNumsNeeded": carEntityNumsNeeded})
    return bch_nnOut_userOut


def transition(userOut: Dict[str, List[str]], cmd: str, unit: str,
               carEntityNums: List[str], carEntityNumsLbl: str, debugg) -> None:
    #assert carEntityNumsLbl and carEntityNums
    if not carEntityNumsLbl or not carEntityNums:
        return
    assert (carEntityNumsLbl == 'year' or carEntityNumsLbl == 'price' or
            carEntityNumsLbl == 'mileage')

    def make_strng(num_s, carEntityNumsLbl, unit):
        return (
         f"{num_s}"
         f"{' ' if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}"
         f"{unit if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}")

    if unit:
        # normalize unit
        if unit in (syntheticData.
                    nonCarEntityLbls_mapTo_entityWrds["units_price"]):
            unit = "$"
        elif unit in (syntheticData.
                      nonCarEntityLbls_mapTo_entityWrds["units_mileage"]):
            if unit.startswith("m"):
                unit = "mi"
            else:
                unit = "km"
        else:
            assert False

    match cmd:
        case 'more' | 'less':
            strng = (
             f"{cmd} {carEntityNums[0]}"
             f"{' ' if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}"
             f"{unit if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}")
            if strng not in userOut[carEntityNumsLbl]:
                userOut[carEntityNumsLbl].append(strng)
            if len(carEntityNums) > 1:
                for carEntityNum in carEntityNums[1:]:
                    strng = make_strng(carEntityNum, carEntityNumsLbl, unit)
                    if strng not in userOut[carEntityNumsLbl]:
                        userOut[carEntityNumsLbl].append(strng)
        case 'range1' | 'range2':
            if len(carEntityNums) < 2:
                return
            strng = (
             f"{carEntityNums[0]}-{carEntityNums[1]}"
             f"{' ' if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}"
             f"{unit if unit and (carEntityNumsLbl in syntheticData.carEntityLbls_require_unit) else ''}")
            if strng not in userOut[carEntityNumsLbl]:
                userOut[carEntityNumsLbl].append(strng)
            if len(carEntityNums) > 2:
                for carEntityNum in carEntityNums[2:]:
                    strng = make_strng(carEntityNum, carEntityNumsLbl, unit)
                    if strng not in userOut[carEntityNumsLbl]:
                        userOut[carEntityNumsLbl].append(strng)
        case _:
            assert not cmd
            if carEntityNums:
                for carEntityNum in carEntityNums:
                    strng = make_strng(carEntityNum, carEntityNumsLbl, unit)
                    if strng not in userOut[carEntityNumsLbl]:
                        userOut[carEntityNumsLbl].append(strng)


def prevTrnUserOut2history(prevTrnUserOut: Dict[str, List[str]]) -> List[str]:
    history: List[str] = []
    for values in prevTrnUserOut.values():
        history.extend(values)
    return history

# ************************** test vectors follow ************************
# python3 -m pdb Utilities.py


#print("start of userIn_filter_splitWords(userIn: str) -> List[str]")
userIns = [
            "- dark-brown, 100.25 - 600.33, 200 -",
            "dark -brown, dark - brown dark-brown dark brown-   red dark#-brown dark#- brown",
            "100 -$200.2, 230 - brown dark-230.5 22 33-   red $22#-34 43#- 55.6",
            "$5000 - $9000",
            "$ 5000 - $ 9000",
            "$5000 - 9000",
            "5000 - $9000",
            "5000$ - 9000$",
            "5000 $ - 9000 $",
            "5000 - 9000$",
            "5000$ - 9000",
            "5000 - 9000",
            "miles 5000 - miles 9000",
            "miles 5000 - 9000",
            "5000 - miles 9000",
            "5000 miles - 9000 miles",
            "5000  - 9000 miles",
            "5000 miles - 9000",
          ]
userIn_filtereds_True = [
        ["dark", "brown", "100.25", "-", "600.33", "200"],
        ["dark", "brown", "dark", "brown", "dark", "brown", "dark", "brown", "red",  "dark", "brown",  "dark", "brown"],
        ["100", "-", "$", "200.2", "230", "brown", "dark", "230.5", "22", "33", "red", "$", "22", "-", "34", "43", "-", "55.6"],
        ["$", "5000", "-", "$", "9000"],
        ["$", "5000", "-", "$", "9000"],
        ["$", "5000", "-", "9000"],
        ["5000", "-", "$", "9000"],
        ["5000", "$", "-", "9000", "$"],
        ["5000", "$", "-", "9000", "$"],
        ["5000", "-", "9000", "$"],
        ["5000", "$", "-", "9000"],
        ["5000", "-", "9000"],
        ["miles", "5000", "-", "miles", "9000"],
        ["miles", "5000", "-", "9000"],
        ["5000", "-", "miles", "9000"],
        ["5000", "miles", "-", "9000", "miles"],
        ["5000", "-", "9000", "miles"],
        ["5000", "miles", "-", "9000"],
       ]
#num_failed = 0
#for userIn, userIn_filtered_True in zip(userIns, userIn_filtereds_True):
#    userIn_filtered = userIn_filter_splitWords(userIn)
#    num_failed = num_failed if (
#            userIn_filtered_True == userIn_filtered) else num_failed+1
#    print(f'userIn  {userIn}\nuserIn_filtered_True {userIn_filtered_True}\n'
#          f'userIn_filtered      {userIn_filtered}\nuserIn_filtered_True == '
#          f'userIn_filtered  {userIn_filtered_True == userIn_filtered}\n')
#print(f'# of failures = {num_failed}')
#print("end of userIn_filter_splitWords(userIn: str) -> List[str]\n\n")


print("start of generate_userOut()")
in_out = [
 [{'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': [], 'year': []},
  ['between',      '$',      '500',    'to',    '600', ],
  ['range1', 'units_price', 'price', 'range2', 'price',],
  {'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': ["500-600 $"], 'year': []}],

 [{'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': ['500-600 $'], 'year': []},
  ['between',      '$',      '500',    'to',    'more', '600',    'dollars',    '1992', 'more', '2001', 'less', '2000',    'miles',          'more'],
  ['range1', 'units_price', 'price', 'range2',  'more', 'price', 'units_price', 'year', 'more', 'year', 'less', 'mileage', 'units_mileage', 'more'],
  {'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': ['more 2000 mi'],
   'price': ['500-600 $', 'more 600 $',], 'year': ['more 1992', 'less 2001',]}],

 [{'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': [], 'year': []},
  ['500',    'to',    '600',   '700',      'mile', ],
  ['price', 'range2', 'price', 'price', 'unit_mileage',],
  {'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': ['500-600', '700'], 'year': []}],

 [{'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': [], 'year': []},
  ['$',            '500',  'less',  '600', ],
  ['units_price', 'price', 'less', 'price',],
  {'brand': [], 'model': [], 'color': [], 'style': [], 'mileage': [],
   'price': ['500 $', 'less 600'], 'year': []}],
]
num_failed = 0
for prev_out, wrds, lbls, out_True in in_out:
    out = generate_userOut([prev_out], [wrds], [lbls])
    num_failed = num_failed if (out[0] == out_True) else num_failed+1
    print(f'wrds  {wrds}\nlbls  {lbls}\nprev_out  {prev_out}\n'
          f'out       {out[0]}\nout_True  {out_True}\n'
          f'out == out_True\t\t {out[0] == out_True}\n')
print(f'# of failures = {num_failed}')
print("end of generate_userOut()")
