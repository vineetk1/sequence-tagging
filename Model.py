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
               batch_size: Dict[str, int]) -> None:
        self.batch_size = batch_size  # needed to turn off lightning warning
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
                 batch_size=self.batch_size['train'],
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
            batch_size=self.batch_size['val'],
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
                 batch_size=self.batch_size['test'],
                 logger=True)
        return ts_loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        pass

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bertModel(**batch['model_inputs'])
        logits = self.classification_head(
            self.classification_head_dropout(outputs[0]))
        loss = self.loss_fct(logits.view(-1, self.num_classes),
                             batch['labels'].view(-1))
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

    def prepare_for_predict(self, dataset_meta: Dict[str, Any],
                            dirPath: pathlib.Path, tokenizer) -> None:
        self.failed_dlgs_file = dirPath.joinpath('dialogs_failed.txt')
        self.failed_dlgs_file.touch()
        if self.failed_dlgs_file.stat().st_size:
            with self.failed_dlgs_file.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')
        self.test_results = dirPath.joinpath('test-results.txt')
        self.test_results.touch()
        if self.test_results.stat().st_size:
            with self.test_results.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')

        self.dataset_meta = dataset_meta
        self.dirPath = dirPath
        self.tokenizer = tokenizer
        self.y_true = []
        self.y_pred = []
        self.df = pd.read_pickle(dataset_meta['dataset_panda'])

    def on_predict_start(self) -> None:
        self.num_of_failed_turns: int = 0
        self.num_of_failed_token_labels: int = 0
        #self.cntr = Counter()
        #self.max_turn_num = 0

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        _, logits = self._run_model(batch)
        predictions = torch.argmax(logits, dim=-1)
        # write to file the info about failed turns of dialogs
        predictions = torch.where(batch['labels'] == -100, batch['labels'],
                                  predictions)
        with self.failed_dlgs_file.open('a') as file:
            prev_failed_dlgTurnIdx = None
            testSet_unseen_tokens = []
            wrapper = textwrap.TextWrapper(width=80,
                                           initial_indent="",
                                           subsequent_indent=19 * " ")
            for failed_dlgTurnIdx, failed_elementIdx in torch.ne(
                    batch['labels'], predictions).nonzero():
                self.num_of_failed_token_labels += 1
                user_inp_tokens_names = self.tokenizer.convert_ids_to_tokens(
                    batch['model_inputs']['input_ids'][failed_dlgTurnIdx])

                true_label_num = batch['labels'][failed_dlgTurnIdx][
                    failed_elementIdx].item()
                pred_label_num = predictions[failed_dlgTurnIdx][
                    failed_elementIdx].item()
                failed_token_label_name_out = (
                    f'{user_inp_tokens_names[failed_elementIdx]}, '
                    f"{self.dataset_meta['token-label names'][true_label_num]}"
                    ', '
                    f"{self.dataset_meta['token-label names'][pred_label_num]}"
                    ';\t')

                if failed_dlgTurnIdx == prev_failed_dlgTurnIdx:
                    file.write(failed_token_label_name_out)
                    continue
                else:
                    self.num_of_failed_turns += 1
                    if testSet_unseen_tokens:
                        strng = ('Test-set tokens not seen in train-set = '
                                 f'{", ".join(testSet_unseen_tokens)}')
                        file.write("\n")
                        file.write(wrapper.fill(strng))
                        testSet_unseen_tokens = []
                if ((batch['model_inputs']['input_ids'][failed_dlgTurnIdx]
                     [failed_elementIdx].item())
                        in self.dataset_meta['test-set unseen tokens']):
                    testSet_unseen_tokens.append(
                        user_inp_tokens_names[failed_elementIdx])
                user_inp_tokens_names_out = "Input tokens = " + " ".join(
                    user_inp_tokens_names)
                dlg_id_out = f"dlg_id = {batch['ids'][failed_dlgTurnIdx]}"
                true_labels_names_out = "True labels = "
                predicted_labels_names_out = "Predicted labels = "
                for i in torch.arange(batch['labels'].shape[1]):
                    if batch['labels'][failed_dlgTurnIdx][i].item() != -100:
                        true_labels_names_out = (
                            true_labels_names_out +
                            self.dataset_meta['token-label names'][
                                batch['labels'][failed_dlgTurnIdx][i].item()] +
                            " ")
                        predicted_labels_names_out = (
                            predicted_labels_names_out +
                            self.dataset_meta['token-label names']
                            [predictions[failed_dlgTurnIdx][i].item()] + " ")
                failed_token_label_name_heading_out = (
                    'Failed token labels: Input token, '
                    'True label, Predicted label; ....')
                file.write("\n\n")
                for strng in (dlg_id_out, user_inp_tokens_names_out,
                              true_labels_names_out,
                              predicted_labels_names_out,
                              failed_token_label_name_heading_out):
                    file.write(wrapper.fill(strng))
                    file.write("\n")
                file.write(failed_token_label_name_out)
                prev_failed_dlgTurnIdx = failed_dlgTurnIdx

            if testSet_unseen_tokens:
                strng = ('Test-set tokens not seen in train-set = '
                         f'{", ".join(testSet_unseen_tokens)}')
                file.write("\n")
                file.write(wrapper.fill(strng))

        # collect info to later calculate precision, recall, f1, etc.
        for prediction, actual in zip(predictions.tolist(),
                                      batch['labels'].tolist()):
            y_true = []
            y_pred = []
            for predicted_token_label_num, actual_token_label_num in zip(
                    prediction, actual):
                if actual_token_label_num != -100:
                    y_true.append(self.dataset_meta['token-label names']
                                  [actual_token_label_num])
                    y_pred.append(self.dataset_meta['token-label names']
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
                    print(f'# of failed turns = {self.num_of_failed_turns}')
                    strng = ('# of failed token-labels = '
                             f'{self.num_of_failed_token_labels}')
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
