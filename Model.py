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
from itertools import zip_longest

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
        self.y_true: List[List[str]] = []
        self.y_pred: List[List[str]] = []
        self.df = pd.read_pickle(dataset_meta['dataset_panda'])

        self.count_failed_turns: int = 0
        self.count_failed_nnOut_tknLblIds: int = 0
        #self.cntr = Counter()
        #self.max_turn_num = 0

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        outputs = self.bertModel(**batch['nnIn_tknIds'])
        logits = self.classification_head(outputs.last_hidden_state)
        bch_nnOut_tknLblIds = torch.argmax(logits, dim=-1)

        bch_userIn_filtered_entityWrds, bch_nnOut_entityWrdLbls = (
            Utilities.tknLbls2entity_wrds_lbls(
                bch_nnIn_tknIds=batch['nnIn_tknIds']['input_ids'],
                bch_map_tknIdx2wrdIdx=batch['map_tknIdx2wrdIdx'],
                bch_userIn_filtered=batch['userIn_filtered'],
                bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
                id2tknLbl=self.dataset_meta['idx2tknLbl'],
                DEBUG_bch_tknLbls_True=batch['tknLblIds'],
                DEBUG_tokenizer=self.tokenizer))

        bch_userOut: List[Dict[str, List[str]]] = (Utilities.generate_userOut(
            bch_userOut=batch['prevTrnUserOut'],
            bch_userIn_filtered_entityWrds=bch_userIn_filtered_entityWrds,
            bch_entityWrdLbls=bch_nnOut_entityWrdLbls))

        if not hasattr(self, 'failed_nnOut_tknLblIds'):
            # predictStatistics is False
            return

        # gather statistics

        # write to file the info about FAILED nnOut_tknLblIds; also keep counts
        # for self.count_failed_turns and self.count_failed_nnOut_tknLblIds
        self._failed_nnOut_tknLblIds(
            bch=batch,
            bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
            bch_userIn_filtered_entityWrds=bch_userIn_filtered_entityWrds,
            bch_nnOut_entityWrdLbls=bch_nnOut_entityWrdLbls,
            bch_userOut=bch_userOut)

        # append new info to self.y_true and self.y_pred; on_predict_end()
        # uses them to calculate precision, recall, f1, etc.
        self._prepare_metric(bch=batch,
                             bch_nnOut_tknLblIds=bch_nnOut_tknLblIds)

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

    def _failed_nnOut_tknLblIds(
            self, bch: Dict[str, Any], bch_nnOut_tknLblIds: torch.Tensor,
            bch_userIn_filtered_entityWrds: List[Union[List[str], None]],
            bch_nnOut_entityWrdLbls: List[Union[List[str], None]],
            bch_userOut: List[Dict[str, List[str]]]) -> None:
        assert bch_nnOut_tknLblIds.shape[0] == len(bch_nnOut_entityWrdLbls)
        bch_nnOut_tknLblIds = torch.where(bch['tknLblIds'] == -100,
                                          bch['tknLblIds'],
                                          bch_nnOut_tknLblIds)
        failed_bchIdxs_nnOutTknLblIdIdxs: List[List[int, int]] = torch.ne(
            bch['tknLblIds'], bch_nnOut_tknLblIds).nonzero().tolist()
        failed_bchIdx: List[int] = [
            failed_bchIdx_nnOutTknLblIdIdx[0] for
            failed_bchIdx_nnOutTknLblIdIdx in failed_bchIdxs_nnOutTknLblIdIdxs
        ]

        bch_entityWrdLbls_True: List[List[str]] = []
        bch_failed_nnOut_entityWrdLbls: List[List[str]] = []
        for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
            bch_entityWrdLbls_True.append([])
            bch_failed_nnOut_entityWrdLbls.append([])
            wrdLbls_True: List[str] = (
                self.df[(self.df['dlgId'] == dlgId)
                        & (self.df['trnId'] == trnId)]['wrdLbls']).item()
            for wrdLbl_True in wrdLbls_True:
                if wrdLbl_True[0] == 'B' or wrdLbl_True[0] == 'I':
                    if wrdLbl_True[-1] == ')':
                        bch_entityWrdLbls_True[-1].append(
                            wrdLbl_True[2:wrdLbl_True.index('(')])
                    else:
                        bch_entityWrdLbls_True[-1].append(wrdLbl_True[2:])
            for entityWrdLbl_True, nnOut_entityWrdLbl in zip_longest(
                    bch_entityWrdLbls_True[-1],
                (bch_nnOut_entityWrdLbls[bch_idx]
                 if bch_nnOut_entityWrdLbls[bch_idx] is not None else [])):
                if entityWrdLbl_True != nnOut_entityWrdLbl:
                    bch_failed_nnOut_entityWrdLbls[-1].append(
                        f"({entityWrdLbl_True}, {nnOut_entityWrdLbl})")
        assert len(bch_entityWrdLbls_True) == len(bch_nnOut_entityWrdLbls)

        bch_userOut_True: List[Dict[str, List[str]]] = []
        bch_failed_nnOut_userOut: List[Union[Dict[str, List[str]], None]] = []
        for bch_idx, (dlgId, trnId) in enumerate(bch['dlgTrnId']):
            bch_userOut_True.append(
                (self.df[(self.df['dlgId'] == dlgId)
                         & (self.df['trnId'] == trnId)]['userOut']).item())
            if bch_userOut[bch_idx] == bch_userOut_True[-1]:
                bch_failed_nnOut_userOut.append(None)
            else:
                d: dict = Utilities.userOut_init()
                for k in d:
                    if bch_userOut[bch_idx][k] != bch_userOut_True[-1][k]:
                        for item_True, item in zip_longest(
                                bch_userOut_True[-1][k],
                                bch_userOut[bch_idx][k]):
                            if item != item_True:
                                d[k].append((item_True, item))
                bch_failed_nnOut_userOut.append(str(d))

        # inner-lists must have same bch_idx occuring consectively
        failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut: List[List[
            int, int]] = []
        for bch_idx in range(len(bch_entityWrdLbls_True)):
            if bch_idx in failed_bchIdx:
                for failed_bchIdx_nnOutTknLblIdIdx in (
                        failed_bchIdxs_nnOutTknLblIdIdxs):
                    if failed_bchIdx_nnOutTknLblIdIdx[0] == bch_idx:
                        failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                            failed_bchIdx_nnOutTknLblIdIdx)
            elif bch_failed_nnOut_entityWrdLbls[bch_idx]:
                failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                    [bch_idx, None])
            elif bch_failed_nnOut_userOut[bch_idx] is not None:
                failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut.append(
                    [bch_idx, None])
            else:
                pass

        if not failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut:
            return

        bch_nnIn_tkns: List[List[str]] = []
        bch_unseen_tkns_predictSet: List[List[str]] = []
        bch_tknLbls_True: List[List[str]] = []
        bch_nnOut_tknLbls: List[List[str]] = []
        nnIn_tknIds_beginEnd_idx = (
            bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
        for bch_idx in range(len(bch_nnOut_entityWrdLbls)):
            bch_nnIn_tkns.append([])
            bch_unseen_tkns_predictSet.append([])
            bch_tknLbls_True.append([])
            bch_nnOut_tknLbls.append([])
            index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[bch_idx * 2,
                                                                1] + 1
            index_of_second_SEP = nnIn_tknIds_beginEnd_idx[(bch_idx * 2) + 1,
                                                           1]
            for nnIn_tknIds_idx in range(index_of_first_SEP_plus1,
                                         index_of_second_SEP):
                nnIn_tknId: int = (bch['nnIn_tknIds']['input_ids'][bch_idx]
                                   [nnIn_tknIds_idx]).item()
                bch_nnIn_tkns[-1].append(
                    self.tokenizer.convert_ids_to_tokens(nnIn_tknId))
                if nnIn_tknId in self.dataset_meta['test-set unseen tokens']:
                    bch_unseen_tkns_predictSet[-1].append(
                        bch_nnIn_tkns[-1][-1])
                bch_tknLbls_True[-1].append(self.dataset_meta['idx2tknLbl'][
                    bch['tknLblIds'][bch_idx, nnIn_tknIds_idx]])
                bch_nnOut_tknLbls[-1].append(self.dataset_meta['idx2tknLbl'][
                    bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx]])

        with self.failed_nnOut_tknLblIds.open('a') as file:
            prev_bch_idx: int = None
            bch_idx: int = None
            wrapper: textwrap.TextWrapper = textwrap.TextWrapper(
                width=80, initial_indent="", subsequent_indent=21 * " ")
            for bch_idx, nnOut_tknLblId_idx in (
                    failed_bchIdxs_nnOutTknLblIdIdxs_entityWrdLbls_userOut):
                # only FAILED bch_idx and nnOut_tknLblId_idx are considered
                self.count_failed_nnOut_tknLblIds += 1
                index_of_first_SEP_plus1 = nnIn_tknIds_beginEnd_idx[bch_idx *
                                                                    2, 1] + 1

                if prev_bch_idx is not None and bch_idx != prev_bch_idx:
                    for strng in (
                            f"entityWrdLbls_True = {' '.join(bch_entityWrdLbls_True[prev_bch_idx])}",
                            f"nnOut_entityWrdLbls = {' '.join(bch_nnOut_entityWrdLbls[prev_bch_idx])}"
                            if bch_nnOut_entityWrdLbls[prev_bch_idx]
                            is not None else "nnOut_entityWrdLbls: None",
                            f"Failed-nnOut_entityWrdLbls (entityWrdLbls_True, nnOut_entityWrdLbls): {', '.join(bch_failed_nnOut_entityWrdLbls[prev_bch_idx])}"
                            if bch_failed_nnOut_entityWrdLbls[prev_bch_idx]
                            else "Failed-nnOut_entityWrdLbls: None",
                            f"userIn_filtered_entityWrds = {' '.join(bch_userIn_filtered_entityWrds[prev_bch_idx])}"
                            if bch_userIn_filtered_entityWrds[prev_bch_idx]
                            is not None else
                            "userIn_filtered_entityWrds: None",
                            f"userOut_True = {bch_userOut_True[prev_bch_idx]}",
                            f"nnOut_userOut = {bch_userOut[prev_bch_idx]}",
                            f"Failed-nnOut_userOut (userOut_True, nnOut_userOut): {bch_failed_nnOut_userOut[prev_bch_idx]}"
                            if bch_failed_nnOut_userOut[prev_bch_idx]
                            is not None else "Failed-nnOut_userOut: None",
                            f"Predict-set tkns not seen in train-set = {', '.join(bch_unseen_tkns_predictSet[bch_idx])}"
                            if bch_unseen_tkns_predictSet[bch_idx] else
                            "Predict-set tkns not seen in train-set: None",
                    ):
                        file.write(wrapper.fill(strng))
                        file.write("\n")

                if bch_idx != prev_bch_idx:
                    # print out: dlgId_trnId, userIn, userIn_filtered wrds,
                    # nnIn_tkns, tknLbls_True, and nnOut_tknLbls;
                    # tknIds between two SEP belong to tknIds of words in
                    # bch['userIn_filtered']
                    self.count_failed_turns += 1
                    file.write("\n\n")
                    for strng in (
                            f"dlg_id, trn_id = {bch['dlgTrnId'][bch_idx]}",
                            f"userIn = {(self.df[(self.df['dlgId'] == bch['dlgTrnId'][bch_idx][0]) & (self.df['trnId'] == bch['dlgTrnId'][bch_idx][1])]['userIn']).item()}",
                            f"userIn_filtered = {' '.join(bch['userIn_filtered'][bch_idx])}",
                            f"nnIn_tkns = {' '.join(bch_nnIn_tkns[bch_idx])}",
                            f"tknLbls_True = {' '.join(bch_tknLbls_True[bch_idx])}",
                            f"nnOut_tknLbls = {' '.join(bch_nnOut_tknLbls[bch_idx])}",
                            "Failed nnOut_tknLbls (userIn_filtered, nnIn_tkn, tknLbl_True, nnOut_tknLbl):"
                            if nnOut_tknLblId_idx is not None else
                            "Failed nnOut_tknLbls: None",
                    ):
                        file.write(wrapper.fill(strng))
                        file.write("\n")

                if nnOut_tknLblId_idx is not None:
                    file.write(
                        wrapper.fill(
                            f"{bch['userIn_filtered'][bch_idx][bch['map_tknIdx2wrdIdx'][bch_idx][nnOut_tknLblId_idx]]}, {bch_nnIn_tkns[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {bch_tknLbls_True[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}, {bch_nnOut_tknLbls[bch_idx][nnOut_tknLblId_idx - index_of_first_SEP_plus1]}  "
                        ))
                    file.write("\n")

                prev_bch_idx = bch_idx

            assert bch_idx is not None
            for strng in (
                    f"entityWrdLbls_True = {' '.join(bch_entityWrdLbls_True[bch_idx])}",
                    f"nnOut_entityWrdLbls = {' '.join(bch_nnOut_entityWrdLbls[bch_idx])}"
                    if bch_nnOut_entityWrdLbls[bch_idx] is not None else
                    "nnOut_entityWrdLbls: None",
                    f"Failed-nnOut_entityWrdLbls (entityWrdLbls_True, nnOut_entityWrdLbls): {', '.join(bch_failed_nnOut_entityWrdLbls[bch_idx])}"
                    if bch_failed_nnOut_entityWrdLbls[bch_idx] else
                    "Failed-nnOut_entityWrdLbls: None",
                    f"userIn_filtered_entityWrds = {' '.join(bch_userIn_filtered_entityWrds[bch_idx])}"
                    if bch_userIn_filtered_entityWrds[bch_idx] is not None else
                    "userIn_filtered_entityWrds: None",
                    f"userOut_True = {bch_userOut_True[prev_bch_idx]}",
                    f"nnOut_userOut = {bch_userOut[prev_bch_idx]}",
                    f"Failed-nnOut_userOut (userOut_True, nnOut_userOut): {bch_failed_nnOut_userOut[prev_bch_idx]}"
                    if bch_failed_nnOut_userOut[prev_bch_idx] is not None else
                    "Failed-nnOut_userOut: None",
                    f"Predict-set tkns not seen in train-set = {' '.join(bch_unseen_tkns_predictSet[bch_idx])}"
                    if bch_unseen_tkns_predictSet[bch_idx] else
                    "Predict-set tkns not seen in train-set: None",
            ):
                file.write(wrapper.fill(strng))
                file.write("\n")

    def _prepare_metric(self, bch: Dict[str, Any],
                        bch_nnOut_tknLblIds: torch.Tensor) -> None:
        assert bch_nnOut_tknLblIds.shape[0] == bch['tknLblIds'].shape[0]
        # tknIds between two SEP belong to tknIds of words in
        # bch['userIn_filtered']
        nnIn_tknIds_idx_beginEnd: torch.Tensor = (
            bch['nnIn_tknIds']['input_ids'] == 102).nonzero()
        for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
            self.y_true.append([])
            self.y_pred.append([])
            prev_firstTknOfWrd_idx: int = None
            for nnIn_tknIds_idx in range(
                (nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1),
                (nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
                if (firstTknOfWrd_idx :=
                        bch['map_tknIdx2wrdIdx'][bch_idx][nnIn_tknIds_idx]
                    ) == prev_firstTknOfWrd_idx:
                    continue  # ignore tknId that is not first-token-of-word
                prev_firstTknOfWrd_idx = firstTknOfWrd_idx
                nnOut_tknLbl_True = self.dataset_meta['idx2tknLbl'][
                    bch['tknLblIds'][bch_idx, nnIn_tknIds_idx]]
                #assert nnOut_tknLbl_True != "T"
                assert nnOut_tknLbl_True[0] != "T"
                self.y_true[-1].append(nnOut_tknLbl_True)
                nnOut_tknLbl = self.dataset_meta['idx2tknLbl'][
                    bch_nnOut_tknLblIds[bch_idx, nnIn_tknIds_idx]]
                #if nnOut_tknLbl == "T":
                if nnOut_tknLbl[0] == "T":
                    # "T" is not allowed in the metric, only BIO is allowed;
                    # the prediction for first-token-in-word must not be "T";
                    # change nnOut_tknLbl so it is not nnOut_tknLbl_True; then
                    # nnOut_tknLbl will be considered a wrong prediction
                    if nnOut_tknLbl_True[0] == "O":
                        nnOut_tknLbl = f"B{nnOut_tknLbl[1:]}"
                    else:
                        nnOut_tknLbl = "O"
                self.y_pred[-1].append(nnOut_tknLbl)
            assert len(self.y_true[-1]) == len(self.y_pred[-1])
        assert len(self.y_true) == len(self.y_pred)
