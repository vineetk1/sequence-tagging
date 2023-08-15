'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from typing import Dict, List, Any, Tuple
import pathlib
from importlib import import_module
import copy
import pandas as pd
from collections import Counter
import math
import Utilities
import Predict_statistics

import os
# disable parallelism in Fast-Tokenizer since it clashes with multiprocessing
# of data-loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logg = getLogger(__name__)


class Model(LightningModule):

    def __init__(self, model_init: dict, num_classes: int,
                 tknLblIds_NumberCount: dict):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model'] == "bert":
            # Dropouts are used ONLY during Training; dropouts are disabled by
            # Lightning during validation or testing or Predicting

            from transformers import BertModel
            self.bertModel = BertModel.from_pretrained(
                model_init['model_type'],
                add_pooling_layer=False,
                hidden_dropout_prob=model_init['hidden_dropout_prob'] if
                (('hidden_dropout_prob' in model_init) and
                 (isinstance(model_init['hidden_dropout_prob'], float))) else
                0.1,
                attention_probs_dropout_prob=model_init[
                    'attention_probs_dropout_prob'] if
                (('attention_probs_dropout_prob' in model_init) and
                 (isinstance(model_init['attention_probs_dropout_prob'],
                             float))) else 0.1,
            )

            self.classification_head_dropout = torch.nn.Dropout(
                model_init['classification_head_dropout'] if
                (('classification_head_dropout' in model_init) and
                 (isinstance(model_init['classification_head_dropout'], float))
                 ) else 0.1)

            self.num_classes = num_classes
            self.classification_head = torch.nn.Linear(
                self.bertModel.config.hidden_size, self.num_classes)

            stdv = 1. / math.sqrt(self.classification_head.weight.size(1))
            self.classification_head.weight.data.uniform_(-stdv, stdv)
            if self.classification_head.bias is not None:
                self.classification_head.bias.data.uniform_(-stdv, stdv)

            if 'loss_func_class_weights' in model_init and isinstance(
                    model_init['loss_func_class_weights'], bool):
                assert self.num_classes == len(tknLblIds_NumberCount)
                assert 0 not in tknLblIds_NumberCount.values()
                # weights with ascending tknLblIds (i.e. classes) without
                # tknLblId=-100 because its Loss is ignored by PyTorch
                weights = torch.tensor([
                    tknLblIds_NumberCount[number]
                    for number in range(self.num_classes)
                ])
                weights = weights / weights.sum()
                weights = 1.0 / weights
                weights = weights / weights.sum()
                self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                self.loss_fct = torch.nn.CrossEntropyLoss()

    def params(self, optz_sched_params: Dict[str, Any],
               bch_sizes: Dict[str, int]) -> None:
        self.bch_sizes = bch_sizes  # needed to turn off lightning warning
        self.optz_sched_params = optz_sched_params
        # Trainer('auto_lr_find': True...) requires self.lr
        self.lr = optz_sched_params['optz_params']['lr'] if (
            'optz_params' in optz_sched_params) and (
                'lr' in optz_sched_params['optz_params']) else None

    def forward(self, batch: Dict[str, Any]):
        outputs = self.bertModel(batch)
        logits = self.classification_head(outputs.last_hidden_state)
        bch_nnOut_tknLblIds = torch.argmax(logits, dim=-1)
        return bch_nnOut_tknLblIds

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        tr_loss, _ = self._run_model(batch)
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 tr_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.bch_sizes['train'],
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
            batch_size=self.bch_sizes['val'],
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
                 batch_size=self.bch_sizes['test'],
                 logger=True)
        return ts_loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        pass

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch['error_msgs']:
            assert False, "Text is longer than allowed"
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
                            dataframes_meta: Dict[str, Any],
                            dirPath: pathlib.Path) -> None:
        if predictStatistics is False:
            self.tokenizer = tokenizer
            self.dataframes_meta: Dict[str, Any] = dataframes_meta
            self.df: pd.DataFrame = pd.read_pickle(
                dataframes_meta['pandas predict-dataframe file location'])
            return

        self.failed_nnOut_tknLblIds_file: pathlib.Path = dirPath.joinpath(
            'failed_nnOut_tknLblIds.txt')
        self.failed_nnOut_tknLblIds_file.touch()
        if self.failed_nnOut_tknLblIds_file.stat().st_size:
            with self.failed_nnOut_tknLblIds_file.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')
        self.passed_file: pathlib.Path = dirPath.joinpath('passed_file.txt')
        self.passed_file.touch()
        if self.passed_file.stat().st_size:
            with self.passed_file.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')
        self.test_results: pathlib.Path = dirPath.joinpath('test-results.txt')
        self.test_results.touch()
        if self.test_results.stat().st_size:
            with self.test_results.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')

        self.tokenizer = tokenizer
        self.dataframes_meta: Dict[str, Any] = dataframes_meta
        self.dirPath: pathlib.Path = dirPath
        self.y_true: List[List[str]] = []
        self.y_pred: List[List[str]] = []
        self.df: pd.DataFrame = pd.read_pickle(
            dataframes_meta['pandas predict-dataframe file location'])

        self.count = {
            # number of turns in Predict dataframe
            "total_turns": 0,
            # number of nnOut_tknLblId that failed; many can fail in same turn
            "failed_tknLbls": 0,
            # number of turns in which nnOut_tknLblIds failed
            "failedTurns_tknLbls": 0,
            # number of turns in which nnOut_entityLbls failed
            "failedTurns_entityLbls": 0,
            # number of turns in which nnOut_userOut failed
            "failedTurns_userOut": 0,
            # num of turns where nnOut_tknLblIds fail but nnOut_entityLbls pass
            "failedTurnsTknLbls_entityLblsPass": 0,
            # num of turns where nnOut_tknLblIds fail but nnOut_userOut pass
            "failedTurnsTknLbls_userOutPass": 0,
            # num of turns where nnOut_tknLblIds pass but nnOut_entityLbls fail
            "passedTurnsTknLbls_entityLblsFail": 0,  # must not happen
            # num of turns where nnOut_tknLblIds pass but nnOut_userOut fail
            "passedTurnsTknLbls_userOutFail": 0,  # must not happen
            # for each tknLbl_True whose nnOut_tknLbls failed, collect a list
            # of passed and failed nnOut_tknLbls and their counts
            "failed_tknLbls_perDlg": {}
        }

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        if batch['error_msgs']:
            assert False, "Text is longer than allowed"
        outputs = self.bertModel(**batch['nnIn_tknIds'])
        logits = self.classification_head(outputs.last_hidden_state)
        bch_nnOut_tknLblIds = torch.argmax(logits, dim=-1)

        bch_nnOut_userIn_filtered_entityWrds, bch_nnOut_entityLbls = (
            Utilities.tknLblIds2entity_wrds_lbls(
                bch_nnIn_tknIds=batch['nnIn_tknIds']['input_ids'],
                bch_map_tknIdx2wrdIdx=batch['map_tknIdx2wrdIdx'],
                bch_userIn_filtered_wrds=batch['userIn_filtered_wrds'],
                bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
                tknLblId2tknLbl=self.dataframes_meta['tknLblId2tknLbl'],
                DEBUG_bch_tknLblIds_True=batch['tknLblIds'],
                DEBUG_tokenizer=self.tokenizer))

        bch_nnOut_userOut: List[Dict[str, List[str]]] = (
            Utilities.generate_userOut(
                # ***NOTE bch_prevTrnUserOut should come from previous turn's
                # bch_nnOut_userOut
                bch_prevTrnUserOut=batch['prevTrnUserOut'],
                bch_nnOut_userIn_filtered_entityWrds=(
                    bch_nnOut_userIn_filtered_entityWrds),
                bch_nnOut_entityLbls=bch_nnOut_entityLbls))

        if not hasattr(self, 'failed_nnOut_tknLblIds_file'):
            # predictStatistics is False
            return

        # gather statistics

        # write to file the info about FAILED nnOut_tknLblIds. etc.; keep count
        Predict_statistics.failed_nnOut_tknLblIds(
            bch=batch,
            bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
            bch_nnOut_userIn_filtered_entityWrds=(
                bch_nnOut_userIn_filtered_entityWrds),
            bch_nnOut_entityLbls=bch_nnOut_entityLbls,
            bch_nnOut_userOut=bch_nnOut_userOut,
            df=self.df,
            tokenizer=self.tokenizer,
            dataframes_meta=self.dataframes_meta,
            count=self.count,
            failed_nnOut_tknLblIds_file=self.failed_nnOut_tknLblIds_file,
            passed_file=self.passed_file,
        )

        # generate self.y_true and self.y_pred; on_predict_end()
        # uses them to calculate precision, recall, f1, etc.
        y_true, y_pred = Predict_statistics.prepare_metric(
            bch=batch,
            bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
            dataframes_meta=self.dataframes_meta)
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def on_predict_end(self) -> None:
        if not hasattr(self, 'failed_nnOut_tknLblIds_file'):
            # predictStatistics is False
            return

        Predict_statistics.print_statistics(
            test_results=self.test_results,
            dataframes_meta=self.dataframes_meta,
            count=self.count,
            y_true=self.y_true,
            y_pred=self.y_pred,
        )
