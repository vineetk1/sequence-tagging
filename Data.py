'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from logging import getLogger
from typing import List, Dict, Any, Tuple
from generate_dataset.Generate_dataset import generate_dataset
from Prepare_dataset_for_trainValTest import prepare_dataset_for_trainValTest
import Utilities

logg = getLogger(__name__)


class Data(LightningDataModule):

    def __init__(self, tokenizer, bch_size: dict):
        super().__init__()
        self.tokenizer = tokenizer
        for bch_size_key in ('train', 'val', 'test', 'predict'):
            if bch_size_key not in bch_size or not isinstance(
                    bch_size[bch_size_key],
                    int) or bch_size[bch_size_key] == 0:
                bch_size[bch_size_key] = 1
        self.bch_sizes = bch_size
        # Trainer('auto_scale_bch_size': True...) requires self.bch_size
        self.bch_size = bch_size['train']

    def generate_dataset(self, dataset_dirPath: str,
                         dataset_split: Dict[str, int]) -> None:
        for dataset_split_key in ('train', 'val', 'test'):
            if dataset_split_key not in dataset_split or not isinstance(
                    dataset_split[dataset_split_key], int):
                dataset_split[dataset_split_key] = 0
        generate_dataset(tokenizer=self.tokenizer,
                         dataset_dirPath=dataset_dirPath,
                         dataset_split=dataset_split)

    def prep_dataset_for_trainValTest(self, dataset_dirPath: str, train: bool,
                                      predict: bool) -> Dict[str, Any]:
        dataset_metadata, train_data, val_data, test_data = (
            prepare_dataset_for_trainValTest(tokenizer=self.tokenizer,
                                             dataset_dirPath=dataset_dirPath,
                                             bch_sizes=self.bch_sizes))
        if train:
            assert (train_data is not None and val_data is not None
                    and test_data is not None)
            self.train_data = Data_set(train_data)
            self.valid_data = Data_set(val_data)
            self.test_data = Data_set(test_data)
        elif predict:
            assert test_data is not None
            self.test_data = Data_set(test_data)
        else:
            strng = 'Train=False and Predict=False; both cannot be False'
            logg.critical(strng)
            exit()
        return dataset_metadata

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.bch_size,
            shuffle=False,
            sampler=RandomSampler(self.train_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            batch_size=self.bch_sizes['val'],
            shuffle=False,
            sampler=RandomSampler(self.valid_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.bch_sizes['test'],
            shuffle=False,
            sampler=RandomSampler(self.test_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.bch_sizes['predict'],
            shuffle=False,
            sampler=RandomSampler(self.test_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def _bert_collater(self,
                       examples: List[List[List[Any]]]) -> Dict[str, Any]:
        bch_dlgTrnId: List[Tuple[int, int]] = []
        bch_userIn_filtered: List[List[str]] = []
        bch_history: List[List[str]] = []
        bch_prevTrnUserOut: List[Dict[str, List[str]]] = []
        map_tknIdx2wrdIdx: List[List[str]] = []

        for example in examples:
            bch_dlgTrnId.append((example[0], example[1]))
            bch_userIn_filtered.append(
                Utilities.userIn_filter_splitWords(example[2]))
            bch_history.append(Utilities.prevTrnUserOut2history(example[3]))
            bch_prevTrnUserOut.append(example[3])

        bch_nnIn_tknIds = self.tokenizer(text=bch_history,
                                         text_pair=bch_userIn_filtered,
                                         is_split_into_words=True,
                                         padding=True,
                                         truncation='do_not_truncate',
                                         return_tensors='pt',
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         return_overflowing_tokens=False)

        for idx in range(len(examples)):
            map_tknIdx2wrdIdx.append(bch_nnIn_tknIds.word_ids(idx))

        # Stop if truncation is needed; Not in Predict
        if bch_nnIn_tknIds['input_ids'].shape[
                1] > self.tokenizer.model_max_length:
            logg.critical('Truncation needed')
            exit()

        # Verify that number of token-ids in history and userIn_filtered are
        # equal to token-label-ids; Not in Predict
        for i, tknLbls_len in enumerate(
                bch_nnIn_tknIds['attention_mask'].count_nonzero(-1)):
            assert tknLbls_len.item() == len(examples[i][4])

        # pad token-label-ids; Not in Predict
        bch_tknLblIds_max_len = max([len(example[4]) for example in examples])
        bch_tknLblIds = torch.LongTensor([
            example[4] + [-100] * (bch_tknLblIds_max_len - len(example[4]))
            for example in examples
        ])

        return {
            'nnIn_tknIds': bch_nnIn_tknIds,  # Training, Predict,
            # Predict-Statistics
            'tknLblIds': bch_tknLblIds,  # Training, Predict,
            # Predict-Statistics
            'dlgTrnId': bch_dlgTrnId,  # Predict-Statistics
            # init_userOut = userOut_init(); bch_userOut =
            # [init_userOut for _ in range(len(examples))] Does NOT work
            # because each copy of dict points to same memory location; i.e.
            # writing a value to a key in a dict will write that value to all
            # dicts
            'prevTrnUserOut': bch_prevTrnUserOut,  # Predict
            'userIn_filtered': bch_userIn_filtered,  # Predict,
            # Predict-Statistics
            'map_tknIdx2wrdIdx': map_tknIdx2wrdIdx,  # Predict
            # Predict-Statistics
        }


class Data_set(Dataset):
    # example = sentence_id plus text plus label
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return (self.examples[idx])
