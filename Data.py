'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from logging import getLogger
from typing import List, Dict, Any
from Generate_dataset import generate_dataset
from Split_dataset import split_dataset

logg = getLogger(__name__)


class Data(LightningDataModule):

    def __init__(self, tokenizer, batch_size: dict):
        super().__init__()
        self.tokenizer = tokenizer
        for batch_size_key in ('train', 'val', 'test', 'predict'):
            if batch_size_key not in batch_size or not isinstance(
                    batch_size[batch_size_key],
                    int) or batch_size[batch_size_key] == 0:
                batch_size[batch_size_key] = 1
        self.batch_size_val = batch_size['val']
        self.batch_size_test = batch_size['test']
        self.batch_size_predict = batch_size['predict']
        # Trainer('auto_scale_batch_size': True...) requires self.batch_size
        self.batch_size = batch_size['train']

    def generate_data_labels(self, dataset_path: str) -> None:
        generate_dataset(tokenize=self.tokenizer, dataset_path=dataset_path)

    def split_dataset(self, dataset_path: str, dataset_split: Dict[str, int],
                      train: bool, predict: bool) -> Dict[str, Any]:
        for dataset_split_key in ('train', 'val', 'test'):
            if dataset_split_key not in dataset_split or not isinstance(
                    dataset_split[dataset_split_key], int):
                dataset_split[dataset_split_key] = 0
        dataset_metadata, train_data, val_data, test_data = split_dataset(
            dataset_path=dataset_path, split=dataset_split)
        dataset_metadata['batch size'] = {
            'train': self.batch_size,
            'val': self.batch_size_val,
            'test': self.batch_size_test,
            'predict': self.batch_size_predict
        }
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
            batch_size=self.batch_size,
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
            batch_size=self.batch_size_val,
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
            batch_size=self.batch_size_test,
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
            batch_size=self.batch_size_predict,
            shuffle=False,
            sampler=RandomSampler(self.test_data),
            batch_sampler=None,
            num_workers=6,
            #num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def _bert_collater(
            self, examples: List[List[List[Any]]]) -> Dict[str, Any]:
        batch_ids, batch_sentences = [], []
        for example in examples:
            batch_ids.append(example[0])
            batch_sentences.append(example[1])

        batch_model_inputs = self.tokenizer(text=batch_sentences,
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt',
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        # Verify that number of tokens in sentence are equal to token-labels
        for i, token_label_len in enumerate(
                batch_model_inputs['attention_mask'].count_nonzero(-1)):
            assert token_label_len.item() == len(examples[i][2])

        # pad token-labels
        batch_token_labels_max_len = max(
            [len(example[2]) for example in examples])
        batch_token_labels = torch.LongTensor([
            example[2] + [-100] *
            (batch_token_labels_max_len - len(example[2]))
            for example in examples
        ])

        return {
            'model_inputs': {
                'input_ids':
                batch_model_inputs['input_ids'].type(torch.LongTensor),
                'attention_mask':
                batch_model_inputs['attention_mask'].type(torch.FloatTensor),
                'token_type_ids':
                batch_model_inputs['token_type_ids'].type(torch.LongTensor)
            },
            'labels': batch_token_labels,
            'ids': tuple(batch_ids)
        }


class Data_set(Dataset):
    # example = sentence_id plus text plus label
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return (self.examples[idx])
