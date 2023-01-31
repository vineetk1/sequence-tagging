'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from typing import Dict, List, Any, Union, Tuple
import pathlib
from importlib import import_module
import copy
import textwrap
import pandas as pd
from collections import Counter
import math
import Utilities
import pickle

import os
# disable parallelism in Fast-Tokenizer since it clashes with multiprocessing
# of data-loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logg = getLogger(__name__)


class Model(LightningModule):

    def __init__(self, model_init: dict, num_classes: int,
                 tokenLabels_NumberCount: dict):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model'] == "bert":
            from transformers import BertModel
            self.num_classes = num_classes
            self.bertModel = BertModel.from_pretrained(
                model_init['model_type'], add_pooling_layer=False)
            self.classification_head_dropout = torch.nn.Dropout(
                model_init['classification_head_dropout'] if
                (('classification_head_dropout' in model_init) and
                 (isinstance(model_init['classification_head_dropout'], float))
                 ) else (self.bertModel.config.hidden_dropout_prob))
            self.classification_head = torch.nn.Linear(
                self.bertModel.config.hidden_size, self.num_classes)
            stdv = 1. / math.sqrt(self.classification_head.weight.size(1))
            self.classification_head.weight.data.uniform_(-stdv, stdv)
            if self.classification_head.bias is not None:
                self.classification_head.bias.data.uniform_(-stdv, stdv)
            if 'loss_func_class_weights' in model_init and isinstance(
                    model_init['loss_func_class_weights'], bool):
                weights = torch.tensor([
                    tokenLabels_NumberCount[number]
                    for number in range(self.num_classes)
                ])
                weights = weights / weights.sum()
                weights = 1.0 / weights
                weights = weights / weights.sum()
                self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                self.loss_fct = torch.nn.CrossEntropyLoss()

    def params(self, optz_sched_params: Dict[str, Any],
               bch_size: Dict[str, int]) -> None:
        self.bch_size = bch_size  # needed to turn off lightning warning
        self.optz_sched_params = optz_sched_params
        # Trainer('auto_lr_find': True...) requires self.lr
        self.lr = optz_sched_params['optz_params']['lr'] if (
            'optz_params' in optz_sched_params) and (
                'lr' in optz_sched_params['optz_params']) else None

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        tr_loss, _ = self._run_model(batch)
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 tr_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.bch_size['train'],
                 logger=False)
        return tr_loss

    def training_epoch_end(
            self, training_step_outputs: List[Dict[str,
                                                   torch.Tensor]]) -> None:
        tr_avg_loss = torch.stack([x['loss']
                                   for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', tr_avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, Any],
                        batch_idx: int) -> torch.Tensor:
        v_loss, _ = self._run_model(batch)
        self.log(
            'val_loss',
            v_loss,
            on_step=False,
            on_epoch=True,  # checkpoint-callback monitors epoch
            # val_loss, so on_epoch Must be True
            prog_bar=True,
            batch_size=self.bch_size['val'],
            logger=False)
        return v_loss

    def validation_epoch_end(self,
                             val_step_outputs: List[torch.Tensor]) -> None:
        v_avg_loss = torch.stack(val_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', v_avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        ts_loss, logits = self._run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('test_loss',
                 ts_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.bch_size['test'],
                 logger=True)
        return ts_loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        pass

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bertModel(**batch['nnIn_tknIds'])
        logits = self.classification_head(
            self.classification_head_dropout(outputs.last_hidden_state))
        loss = self.loss_fct(logits.view(-1, self.num_classes),
                             batch['tknLblIds'].view(-1))
        return loss, logits

    def configure_optimizers(self):
        opt_sch_params = copy.deepcopy(self.optz_sched_params)
        _ = opt_sch_params['optz_params'].pop('lr', None)
        if 'optz' in opt_sch_params and opt_sch_params['optz']:
            if 'optz_params' in opt_sch_params and opt_sch_params[
                    'optz_params']:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            lr=self.lr,
                                            **opt_sch_params['optz_params'])
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            **opt_sch_params['optz_params'])
            else:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(), lr=self.lr)
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters())

        if 'lr_sched' in opt_sch_params and opt_sch_params['lr_sched']:
            if 'lr_sched_params' in opt_sch_params and opt_sch_params[
                    'lr_sched_params']:
                scheduler = getattr(import_module('torch.optim.lr_scheduler'),
                                    opt_sch_params['lr_sched'])(
                                        optimizer=optimizer,
                                        **opt_sch_params['lr_sched_params'])
            else:
                scheduler = getattr(
                    import_module('torch.optim.lr_scheduler'),
                    opt_sch_params['lr_sched'])(optimizer=optimizer)

        # If scheduler is specified then optimizer must be specified
        # If Trainer('resume_from_checkpoint',...), then optimizer and
        # scheduler may not be specified
        if 'optimizer' in locals() and 'scheduler' in locals():
            return {
                'optimizer':
                optimizer,
                'lr_scheduler':
                scheduler,
                'monitor':
                'val_loss'
                if opt_sch_params['lr_sched'] == 'ReduceLROnPlateau' else None
            }
        elif 'optimizer' in locals():
            return optimizer

    def prepare_for_predict(self, predictStatistics: bool, tokenizer,
                            dataset_meta: Dict[str, Any],
                            dirPath: pathlib.Path) -> None:
        if predictStatistics is False:
            # tokenizer and df are needed for debugging ONLY by
            # Utilities.DEBUG_tknLbls2entity_wrds_lbls()
            self.tokenizer = tokenizer
            self.dataset_meta = dataset_meta
            self.df = pd.read_pickle(dataset_meta['dataset_panda'])
            return

        self.failed_nnOut_tknLblIds = dirPath.joinpath(
            'failed_nnOut_tknLblIds.txt')
        self.failed_nnOut_tknLblIds.touch()
        if self.failed_nnOut_tknLblIds.stat().st_size:
            with self.failed_nnOut_tknLblIds.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')
        self.test_results = dirPath.joinpath('test-results.txt')
        self.test_results.touch()
        if self.test_results.stat().st_size:
            with self.test_results.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')

        self.tokenizer = tokenizer
        self.dataset_meta = dataset_meta
        self.dirPath = dirPath
        self.y_true = []
        self.y_pred = []
        self.df = pd.read_pickle(dataset_meta['dataset_panda'])

        self.count_failed_turns: int = 0
        self.count_failed_nnOut_tknLblIds: int = 0
        #self.cntr = Counter()
        #self.max_turn_num = 0

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        outputs = self.bertModel(**batch['nnIn_tknIds'])
        logits = self.classification_head(outputs.last_hidden_state)
        bch_nnOut_tknLblIds = torch.argmax(logits, dim=-1)

        #bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls = (
        #    Utilities.DEBUG_tknLbls2entity_wrds_lbls(
        #        bch=batch,
        #        bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
        #        id2tknLbl=self.dataset_meta['idx2tknLbl'],
        #        tokenizer=self.tokenizer,
        #        df=self.df,))

        bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls = (
            Utilities.tknLbls2entity_wrds_lbls(
                bch=batch,
                bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
                id2tknLbl=self.dataset_meta['idx2tknLbl']))

        bch_userOut: List[Dict[str, List[str]]] = Utilities.generate_userOut(
            bch_userOut=batch['prevTrnUserOut'],
            bch_userIn_filtered_entityWrds=bch_userIn_filtered_entityWrds,
            bch_entityWrdLbls=bch_nnOut_entityWrdLbls)

        if not hasattr(self, 'failed_nnOut_tknLblIds'):
            # predictStatistics is False
            return

        # gather statistics

        # write to file the info about FAILED nnOut_tknLblIds
        self._failed_nnOut_tknLblIds(bch=batch,
                                     bch_nnOut_tknLblIds=bch_nnOut_tknLblIds)

        # collect info to later calculate precision, recall, f1, etc.
        for prediction, actual in zip(bch_nnOut_tknLblIds.tolist(),
                                      batch['tknLblIds'].tolist()):
            y_true = []
            y_pred = []
            for predicted_token_label_num, actual_token_label_num in zip(
                    prediction, actual):
                if actual_token_label_num != -100:
                    y_true.append(self.dataset_meta['idx2tknLbl']
                                  [actual_token_label_num])
                    y_pred.append(self.dataset_meta['idx2tknLbl']
                                  [predicted_token_label_num])
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)
            assert len(y_true) == len(y_pred)

    def on_predict_end(self) -> None:
        # Print
        from sys import stdout
        from contextlib import redirect_stdout
        from pathlib import Path
        stdoutput = Path('/dev/null')
        for out in (stdoutput, self.test_results):
            with out.open("a") as results_file:
                with redirect_stdout(stdout if out ==
                                     stdoutput else results_file):
                    print(f'# of failed turns = {self.count_failed_turns}')
                    strng = ('# of failed token-labels = '
                             f'{self.count_failed_nnOut_tknLblIds}')
                    print(strng)
                    for k, v in self.dataset_meta.items():
                        print(k)
                        print(
                            textwrap.fill(f'{v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))

                    from seqeval.scheme import IOB2
                    from seqeval.metrics import accuracy_score
                    from seqeval.metrics import precision_score
                    from seqeval.metrics import recall_score
                    from seqeval.metrics import f1_score
                    from seqeval.metrics import classification_report
                    print('Classification Report')
                    print(
                        classification_report(self.y_true,
                                              self.y_pred,
                                              mode='strict',
                                              scheme=IOB2))
                    print('Precision = ', end="")
                    print(
                        precision_score(self.y_true,
                                        self.y_pred,
                                        mode='strict',
                                        scheme=IOB2))
                    print('Recall = ', end="")
                    print(
                        recall_score(self.y_true,
                                     self.y_pred,
                                     mode='strict',
                                     scheme=IOB2))
                    print('F1 = ', end="")
                    print(
                        f1_score(self.y_true,
                                 self.y_pred,
                                 mode='strict',
                                 scheme=IOB2))
                    strng = (
                        'Accuracy = '
                        f'{accuracy_score(self.y_true, self.y_pred): .2f}')
                    print(strng)

    def _failed_nnOut_tknLblIds(self, bch: Dict[str, Any],
                                bch_nnOut_tknLblIds: torch.Tensor) -> None:
        bch_nnOut_tknLblIds = torch.where(bch['tknLblIds'] == -100,
                                          bch['tknLblIds'],
                                          bch_nnOut_tknLblIds)
        with self.failed_nnOut_tknLblIds.open('a') as file:
            prev_bch_idx: int = None
            bch_idx: int = None
            unseen_tkns_predictSet: List[str] = []
            tknLbls_True: List[str] = []
            nnOut_tknLbls: List[str] = []
            wrapper: textwrap.TextWrapper = textwrap.TextWrapper(
                width=80, initial_indent="", subsequent_indent=23 * " ")
            for bch_idx, nnOut_tknLblId_idx in torch.ne(
                    bch['tknLblIds'], bch_nnOut_tknLblIds).nonzero():
                # only FAILED bch_idx and nnOut_tknLblId_idx are considered
                self.count_failed_nnOut_tknLblIds += 1
                id2tknLbl = self.dataset_meta['idx2tknLbl']

                if prev_bch_idx is not None and bch_idx != prev_bch_idx:
                    if unseen_tkns_predictSet:
                        for strng in (
                                "Predict-set tkns not seen in train-set:",
                                f"{' '.join(unseen_tkns_predictSet)}",
                        ):
                            file.write(wrapper.fill(strng))
                            file.write("\n")
                        unseen_tkns_predictSet.clear()

                if bch_idx != prev_bch_idx:
                    # print out: dlgId_trnId, userIn, userIn_filtered wrds,
                    # nnIn_tkns, tknLbls_True, and nnOut_tknLbls;
                    # tknIds between two SEP belong to tknIds of words in
                    # bch['userIn_filtered']
                    self.count_failed_turns += 1
                    nnIn_tknIds_beginEnd_idx = (
                        bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
                    index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[
                        bch_idx * 2, 1] + 1
                    index_of_second_SEP = nnIn_tknIds_beginEnd_idx[(bch_idx *
                                                                    2) + 1, 1]
                    nnIn_tknIds: torch.Tensor = bch['nnIn_tknIds'][
                        'input_ids'][bch_idx][
                            index_of_first_SEP_plus1:index_of_second_SEP]
                    unseen_tkns_predictSet = [
                        self.tokenizer.convert_ids_to_tokens(nnIn_tknId)
                        for nnIn_tknId in nnIn_tknIds if nnIn_tknId in
                        self.dataset_meta['test-set unseen tokens']
                    ]
                    nnIn_tkns: List[str] = (
                        self.tokenizer.convert_ids_to_tokens(nnIn_tknIds))
                    tknLbls_True.clear()
                    nnOut_tknLbls.clear()
                    for nnIn_tknIds_idx in range(index_of_first_SEP_plus1,
                                                 index_of_second_SEP):
                        tknLbls_True.append(
                            id2tknLbl[bch['tknLblIds'][bch_idx,
                                                       nnIn_tknIds_idx]])
                        nnOut_tknLbls.append(
                            id2tknLbl[bch_nnOut_tknLblIds[bch_idx,
                                                          nnIn_tknIds_idx]])
                    file.write("\n\n")
                    for strng in (
                            f"dlg_id, trn_id = {bch['dlgTrnId'][bch_idx]}",
                            f"userIn = {(self.df[(self.df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (self.df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userIn']).item()}",
                            f"userIn_filtered wrds = {' '.join(bch['userIn_filtered'][bch_idx])}",
                            f"nnIn_tkns = {' '.join(nnIn_tkns)}",
                            f"tknLbls_True = {' '.join(tknLbls_True)}",
                            f"nnOut_tknLbls = {' '.join(nnOut_tknLbls)}",
                            "Failed nnOut_tknLbls:",
                            "(userIn_filtered wrd, nnIn_tkns, tknLbl_True, nnOut_tknLbl)",
                    ):
                        file.write(wrapper.fill(strng))
                        file.write("\n")

                file.write(
                    wrapper.fill(
                        f"({bch['userIn_filtered'][bch_idx][bch['map_tknIdx2wrdIdx'][bch_idx][nnOut_tknLblId_idx]]}, {nnIn_tkns[nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {tknLbls_True[nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {nnOut_tknLbls[nnOut_tknLblId_idx - index_of_first_SEP_plus1]})  "
                    ))
                file.write("\n")

                prev_bch_idx = bch_idx

            if unseen_tkns_predictSet:
                for strng in (
                        "Predict-set tkns not seen in train-set:",
                        f"{' '.join(unseen_tkns_predictSet)}",
                ):
                    file.write(wrapper.fill(strng))
                    file.write("\n")
