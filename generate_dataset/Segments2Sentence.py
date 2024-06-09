"""
Vineet Kumar, xyoom.ai
"""

from logging import getLogger
from typing import List, Dict, Union
import random
from generate_dataset.Synthetic_dataset import (full_sentences,
                                                segments_weights)

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
        for entityLbl, segs_wgts in segments_weights.items():
            segments = list(segs_wgts["segments"])
            random.shuffle(segments)
            self.segmentsInfo_per_entityLbl[entityLbl] = {
                'segs': segments,
                'seg_idx': 0,
                'segs_used': 0,
            }

    def get_sentence(self, segs_per_sentence=None) -> str:
        if not self.full_sentences['sentences']:
            self.full_sentences['sentences_used'] = True
        if not self.full_sentences['sentences_used']:
            sentence = self.full_sentences['sentences'][
                self.full_sentences['sentence_idx']]
            if self.full_sentences['sentence_idx'] == (
                    len(self.full_sentences['sentences']) - 1):
                self.full_sentences['sentences_used'] = True
            else:
                self.full_sentences['sentence_idx'] += 1
            return sentence

        segments_per_sentence = random.randint(
            1, len(self.segmentsInfo_per_entityLbl))
        seg_names: List[str] = list(segments_weights.keys())
        weights: List[int] = []
        for seg_wgt in segments_weights.values():
            weights.append(seg_wgt["weight"])
        sentence: str = ""
        segmentParts_count: int = 0
        while segments_per_sentence:
            entityLbl = random.choices(seg_names, weights=weights, k=1)[0]
            segment = self.segmentsInfo_per_entityLbl[entityLbl]['segs'][
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx']]
            # if tokenizer.model_max_length = 512 then user_in <= 100 words
            # Assume each segment-part (e.g. <brand><> has 1 word,
            # <multilabel><> has 2 words, <other><> has upto 3 words) generates
            # an average of 1.5 words, then there must be a max of 67 (100/1.5)
            # segment-parts in a sentence
            seg_parts = len(segment.split())
            assert seg_parts <= 67, "More than 67 segment-parts in a segment"
            if segmentParts_count + seg_parts <= 67:
                sentence = f"{sentence}{' ' if sentence else ''}{segment}"
                segmentParts_count = segmentParts_count + seg_parts
            else:
                break
            if self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] == (
                    len(self.segmentsInfo_per_entityLbl[entityLbl]['segs']) -
                    1):
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] = 0
                self.segmentsInfo_per_entityLbl[entityLbl]['segs_used'] += 1
                random.shuffle(
                    self.segmentsInfo_per_entityLbl[entityLbl]['segs'])
            else:
                self.segmentsInfo_per_entityLbl[entityLbl]['seg_idx'] += 1
            segments_per_sentence -= 1
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
        min_numOf_segs_created = 1500
        segs_used: bool = True
        for segmentsInfo in self.segmentsInfo_per_entityLbl.values():
            if not (segs_used and
                    (segmentsInfo['segs_used'] * len(segmentsInfo['segs']) >=
                     min_numOf_segs_created)):
                return False
        return True
