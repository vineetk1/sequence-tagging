"""
Vineet Kumar, xyoom.ai
"""

from logging import getLogger
from typing import List, Dict, Any, Tuple
import torch
from Model import Model
import Utilities
from transformers import BertTokenizerFast
from transformers import BertModel
import pathlib
import pickle

logg = getLogger(__name__)


class Pipeline():

    def __init__(self):
        super().__init__()
        dataframes_dirPath: pathlib.Path = pathlib.Path(
            'experiments/14').resolve(strict=True)
        df_metadata_file: pathlib.Path = dataframes_dirPath.joinpath(
            'df_metadata')
        with df_metadata_file.open('rb') as file:
            self.dataframes_meta: Dict[str, Any] = pickle.load(file)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-large-uncased',
            map_location=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.tokenizer.truncation_side = 'right'  # this is the default also
        # comment the line below because I want the default value of 512
        # token-ids for the  max length of input to the model
        # self.tokenizer.model_max_length = 100

        # following line has dependence on Lightning
        self.model = Model.load_from_checkpoint(
            # ******* NOTE: when checkpoint changes, also change dataframes_dirPath *******
            '/home/vin/sequence-tagging/experiments/14/model=bert,model_type=bert-large-uncased,tokenizer_type=bert/ckpts_v0/checkpoints/lr_sched=ReduceLROnPlateau,factor=0.5,mode=min,patience=4,optz=Adam,lr=1e-05,epoch=17-val_loss=0.01394.ckpt',
            map_location=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"))
        #self.model = torch.quantization.quantize_dynamic(self.model,
        #                                                 {torch.nn.Linear},
        #                                                 dtype=torch.qint8)

    def input(self, sessionId: str, userIn: str,
              prevTrnUserOut: Dict[str, List[str]]):
        # len(userIn) is greater than 0; rest of code accept
        # function arguments in batches
        batch: Dict[str, Any] = self._bert_tokenize(
            [[sessionId, userIn, prevTrnUserOut]])
        bch_nnOut_userOut: List[Dict[str, List[str]]] = self._forward(batch)
        print(f"bch_nnOut_userOut= {bch_nnOut_userOut}\n")
        return bch_nnOut_userOut[0]

    def _bert_tokenize(self, examples: List[List[Any]]) -> Dict[str, Any]:
        bch_dlgTrnId: List[Tuple[int, int]] = []
        bch_userIn_filtered_wrds: List[List[str]] = []
        bch_history: List[List[str]] = []
        bch_prevTrnUserOut: List[Dict[str, List[str]]] = []
        map_tknIdx2wrdIdx: List[List[str]] = []

        for example in examples:
            bch_dlgTrnId.append((example[0], 0))
            print(f"userIn= {example[1]}")
            bch_userIn_filtered_wrds.append(
                Utilities.userIn_filter_splitWords(example[1]))
            print(f"bch_userIn_filtered_wrds= {bch_userIn_filtered_wrds}")
            if len(example[2]):
                bch_prevTrnUserOut.append(example[2])
            else:
                bch_prevTrnUserOut.append(Utilities.userOut_init())
            bch_history.append(
                Utilities.prevTrnUserOut2history(bch_prevTrnUserOut[-1]))
            if (len(bch_history[-1]) * 3) + len(
                    bch_userIn_filtered_wrds[-1]) > 101:
                # history is generated from prevTrnUserOut; In file
                # Generate_dataframes.py, history became empty list and then
                # tknLblIds were generated; here also history becomes empty
                # list and later tknLblIds are generated
                bch_history[-1] = []

        # return_attention_mask must be True for the model to work properly
        bch_nnIn_tknIds = self.tokenizer(
            text=bch_history,
            text_pair=bch_userIn_filtered_wrds,
            is_split_into_words=True,
            padding=True,
            truncation='only_second',
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_overflowing_tokens=False).to(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # ".to(..." is not needed when running on huggingface-spaces

        for idx in range(len(examples)):
            map_tknIdx2wrdIdx.append(bch_nnIn_tknIds.word_ids(idx))

        return {
            'nnIn_tknIds': bch_nnIn_tknIds,
            'dlgTrnId': bch_dlgTrnId,
            'prevTrnUserOut': bch_prevTrnUserOut,
            'userIn_filtered_wrds': bch_userIn_filtered_wrds,
            'map_tknIdx2wrdIdx': map_tknIdx2wrdIdx,
        }

    def _forward(self, batch: Dict[str, Any]) -> List[Dict[str, List[str]]]:
        self.model.eval()
        with torch.no_grad():
            # following line has dependence on Lightning
            bch_nnOut_tknLblIds: torch.Tensor = self.model(
                batch['nnIn_tknIds'])  # model.forward()
        D_bch_nnIn_tkns, D_bch_nnOut_tknLbls = self._convert_ids2tkns(
            batch['nnIn_tknIds']['input_ids'], bch_nnOut_tknLblIds)
        print("(nnIn_tkn, nnOUt_tknLbl)= ", end="")
        for nnIn_tkn, nnOut_tknLbl in zip(D_bch_nnIn_tkns,
                                          D_bch_nnOut_tknLbls):
            print(f"({nnIn_tkn}, {nnOut_tknLbl}), ", end="")
        print()

        bch_nnOut_userIn_filtered_entityWrds, bch_nnOut_entityLbls = (
            Utilities.tknLblIds2entity_wrds_lbls(
                bch_nnIn_tknIds=batch['nnIn_tknIds']['input_ids'],
                bch_map_tknIdx2wrdIdx=batch['map_tknIdx2wrdIdx'],
                bch_userIn_filtered_wrds=batch['userIn_filtered_wrds'],
                bch_nnOut_tknLblIds=bch_nnOut_tknLblIds,
                tknLblId2tknLbl=self.dataframes_meta['tknLblId2tknLbl'],
            ))
        print(
            f"bch_nnOut_userIn_filtered_entityWrds= {bch_nnOut_userIn_filtered_entityWrds}"
        )
        print(f"bch_nnOut_entityLbls= {bch_nnOut_entityLbls}")

        bch_nnOut_userOut: List[Dict[str, List[str]]] = (
            Utilities.generate_userOut(
                bch_prevTrnUserOut=batch['prevTrnUserOut'],
                bch_nnOut_userIn_filtered_entityWrds=(
                    bch_nnOut_userIn_filtered_entityWrds),
                bch_nnOut_entityLbls=bch_nnOut_entityLbls))
        return bch_nnOut_userOut

    def _convert_ids2tkns(self, bch_nnIn_tknIds, bch_nnOut_tknLblIds):
        nnIn_tknIds_idx_beginEnd: torch.Tensor = (
            bch_nnIn_tknIds == 102).nonzero()
        assert bch_nnIn_tknIds.shape[0] * 2 == nnIn_tknIds_idx_beginEnd.shape[
            0]

        D_nnIn_tkn = []
        D_nnOut_tknLbl = []
        for bch_idx in range(bch_nnOut_tknLblIds.shape[0]):
            #for D_nnIn_tknIds_idx in range(bch_nnOut_tknLblIds.shape[1]):
            for D_nnIn_tknIds_idx in range(
                (nnIn_tknIds_idx_beginEnd[bch_idx * 2, 1] + 1),
                (nnIn_tknIds_idx_beginEnd[(bch_idx * 2) + 1, 1])):
                D_nnIn_tkn.append(
                    self.tokenizer.convert_ids_to_tokens(
                        bch_nnIn_tknIds[bch_idx][D_nnIn_tknIds_idx].item()))
                D_nnOut_tknLbl.append(self.dataframes_meta['tknLblId2tknLbl'][
                    bch_nnOut_tknLblIds[bch_idx, D_nnIn_tknIds_idx]])
        return D_nnIn_tkn, D_nnOut_tknLbl
