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
        self.batch_sizes = batch_size
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
            dataset_path=dataset_path,
            splits=dataset_split,
            batch_sizes=self.batch_sizes)
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
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_sizes['val'],
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
            batch_size=self.batch_sizes['test'],
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
            batch_size=self.batch_sizes['predict'],
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
        batch_ids, batch_histories, batch_sentences = [], [], []
        for example in examples:
            batch_ids.append((example[0], example[1]))
            batch_histories.append(example[2][0])
            batch_sentences.append(example[2][1])

        batch_model_inputs = self.tokenizer(text=batch_histories,
                                            text_pair=batch_sentences,
                                            padding=True,
                                            truncation='only_first',
                                            return_tensors='pt',
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            return_overflowing_tokens=False)

        # Verify that number of tokens in sentence are equal to token-labels;
        # Not in Deployment
        for i, token_label_len in enumerate(
                batch_model_inputs['attention_mask'].count_nonzero(-1)):
            assert token_label_len.item() == len(examples[i][3])

        """
        indices_of_sep = torch.nonzero(
            batch_model_inputs['input_ids'] == self.tokenizer.sep_token_id)
        assert (indices_of_sep.size()[0] ==
                batch_model_inputs['input_ids'].size()[0] * 2)
        for i in range(batch_model_inputs['input_ids'].size()[0] * 2):
            j = i * 2 if i else i
            assert (indices_of_sep[j + 1, 1] - indices_of_sep[j, 1] +
                    1) == len(examples[i][3])
        """

        # pad token-labels; Not in Deployment
        batch_token_labels_max_len = max(
            [len(example[3]) for example in examples])
        batch_token_labels = torch.LongTensor([
            example[3] + [-100] *
            (batch_token_labels_max_len - len(example[3]))
            for example in examples
        ])

        return {
            'model_inputs': {
                'input_ids':
                batch_model_inputs['input_ids'].type(torch.LongTensor),
                'attention_mask':
                batch_model_inputs['attention_mask'].type(torch.FloatTensor),
            },
            'labels': batch_token_labels,
            'ids': (batch_ids)
        }


class Data_set(Dataset):
    # example = sentence_id plus text plus label
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return (self.examples[idx])
