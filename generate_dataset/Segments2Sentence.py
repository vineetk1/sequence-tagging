'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Union
import random
from generate_dataset.Synthetic_dataset import (full_sentences,
                                                entityLbls_mapTo_segments)

logg = getLogger(__name__)


class Segments2Sentence():

    def __init__(self, trainValTest):
        self.full_sentences = {
            'sentences': list(full_sentences[trainValTest]),
            'sentence_idx': 0,
            'sentences_used': False
        }
        random.shuffle(self.full_sentences['sentences'])

        self.segmentsInfo_per_entityLbl: Dict[str, Dict[str,
                                                        Union[List[str], int,
                                                              bool]]] = {}
        for entityLbl, segments in entityLbls_mapTo_segments.items():
            segments = list(segments)
            random.shuffle(segments)
            self.segmentsInfo_per_entityLbl[entityLbl] = {
                'segs': segments,
                'seg_idx': 0,
                'segs_used': 0,
            }

    def get_sentence(self, segs_per_sentence=None) -> str:
        if not self.full_sentences['sentences_used']:
            sentence = self.full_sentences['sentences'][
                self.full_sentences['sentence_idx']]
            if self.full_sentences['sentence_idx'] == (
                    len(self.full_sentences['sentences']) - 1):
                self.full_sentences['sentences_used'] = True
            else:
                self.full_sentences['sentence_idx'] += 1
            return sentence
        segments_per_sentence = (
            random.randint(1, len(self.segmentsInfo_per_entityLbl))
            if segs_per_sentence is None else segs_per_sentence)
        random_entityLbls_of_segmentsInfo_per_entityLbl = random.choices(
            list(self.segmentsInfo_per_entityLbl.keys()),
            k=segments_per_sentence)
        sentence = ""
        for entityLbl in random_entityLbls_of_segmentsInfo_per_entityLbl:
            segment = self.segmentsInfo_per_entityLbl[entityLbl]['segs'][
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx']]
            sentence = f"{sentence}{' ' if sentence else ''}{segment}"
            if self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] == (
                    len(self.segmentsInfo_per_entityLbl[entityLbl]['segs']) -
                    1):
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] = 0
                self.segmentsInfo_per_entityLbl[entityLbl]['segs_used'] += 1
                random.shuffle(
                    self.segmentsInfo_per_entityLbl[entityLbl]['segs'])
            else:
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] += 1
        return sentence

    def get_segment(self) -> str:
        while not self.full_sentences['sentences_used']:
            sentence = self.full_sentences['sentences'][
                self.full_sentences['sentence_idx']]
            if self.full_sentences['sentence_idx'] == (
                    len(self.full_sentences['sentences']) - 1):
                self.full_sentences['sentences_used'] = True
            else:
                self.full_sentences['sentence_idx'] += 1
            yield sentence

        for segmentsInfo in self.segmentsInfo_per_entityLbl.values():
            for seg_idx in range(len(segmentsInfo['segs'])):
                yield segmentsInfo['segs'][seg_idx]
        for segmentsInfo in self.segmentsInfo_per_entityLbl.values():
            segmentsInfo['segs_used'] += 1

    def all_segments_done(self) -> bool:
        numOfTimes_segs_used = 1000
        segs_used: bool = True
        for segmentsInfo in self.segmentsInfo_per_entityLbl.values():
            segs_used = segs_used and (segmentsInfo['segs_used'] >=
                                       numOfTimes_segs_used)
            if not segs_used:
                return False
        return True
