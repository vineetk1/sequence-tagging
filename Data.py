'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from logging import getLogger
from typing import List, Dict, Any, Tuple
from Generate_dataset import generate_dataset
from Split_dataset import split_dataset
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

    def generate_data_labels(self, dataset_path: str) -> None:
        generate_dataset(tokenize=self.tokenizer, dataset_path=dataset_path)

    def split_dataset(self, dataset_path: str, dataset_split: Dict[str, int],
                      train: bool, predict: bool) -> Dict[str, Any]:
        for dataset_split_key in ('train', 'val', 'test'):
            if dataset_split_key not in dataset_split or not isinstance(
                    dataset_split[dataset_split_key], int):
                dataset_split[dataset_split_key] = 0
        dataset_metadata, train_data, val_data, test_data = split_dataset(
            tokenizer=self.tokenizer,
            dataset_path=dataset_path,
            splits=dataset_split,
            bch_sizes=self.bch_sizes)
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
            num_workers=6,
            #num_workers=0,
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
            num_workers=6,
            #num_workers=0,
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
            num_workers=6,
            #num_workers=0,
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
            num_workers=6,
            #num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def _bert_collater(self,
                       examples: List[List[List[Any]]]) -> Dict[str, Any]:
        bch_dlgTrn_ids: List[Tuple[int, int]] = []
        bch_userIn_filtered_wrds: List[List[str]] = []
        bch_history: List[List[str]] = []
        for example in examples:
            bch_dlgTrn_ids.append((example[0], example[1]))
            bch_userIn_filtered_wrds.append(
                Utilities.userIn_filter_splitWords(example[2]))
            bch_history.append(example[3])

        bch_nnIn_tknIds = self.tokenizer(text=bch_history,
                                         text_pair=bch_userIn_filtered_wrds,
                                         is_split_into_words=True,
                                         padding=True,
                                         truncation='only_first',
                                         return_tensors='pt',
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         return_overflowing_tokens=False)

        # Verify that number of tokens in history and userIn are equal to
        # token-labels; Not in Deployment
        for i, tknLbls_len in enumerate(
                bch_nnIn_tknIds['attention_mask'].count_nonzero(-1)):
            assert tknLbls_len.item() == len(examples[i][4])

        # pad token-labels; Not in Deployment
        bch_tknLbls_max_len = max([len(example[4]) for example in examples])
        bch_tknLbls = torch.LongTensor([
            example[4] + [-100] * (bch_tknLbls_max_len - len(example[4]))
            for example in examples
        ])

        return {
            'userIn_filtered_wrds': bch_userIn_filtered_wrds,
            'nnIn_tknIds': bch_nnIn_tknIds,
            'dlgTrn_id': bch_dlgTrn_ids,
            'tknLbl_ids': bch_tknLbls,
            'userOut': len(examples) * [Utilities.userOut_init()]
        }


class Data_set(Dataset):
    # example = sentence_id plus text plus label
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return (self.examples[idx])
