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


class Inference():

    def __init__(self):
        super().__init__()
        dataframes_dirPath: pathlib.Path = pathlib.Path(
            'experiments/4').resolve(strict=True)
        df_metadata_file: pathlib.Path = dataframes_dirPath.joinpath(
            'df_metadata')
        with df_metadata_file.open('rb') as file:
            self.dataframes_meta: Dict[str, Any] = pickle.load(file)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-large-uncased')
        self.tokenizer.truncation_side = 'right'  # this is the default also
        # comment the line below because I want the default value of 512
        # token-ids for the  max length of input to the model
        # self.tokenizer.model_max_length = 100

        # following line has dependence on Lightning
        self.model = Model.load_from_checkpoint(
        #    '/home/vin/sequence-tagging/experiments/4/model=bert,model_type=bert-large-uncased,tokenizer_type=bert/ckpts_v0/checkpoints/lr_sched=ReduceLROnPlateau,factor=0.5,mode=min,patience=4,optz=Adam,lr=1e-05,epoch=12-val_loss=0.01277.ckpt'
        #    '/home/vin/sequence-tagging/experiments/4/model=bert,model_type=bert-large-uncased,tokenizer_type=bert/ckpts_v0/checkpoints/lr_sched=ReduceLROnPlateau,factor=0.5,mode=min,patience=4,optz=Adam,lr=1e-05,epoch=20-val_loss=0.01791.ckpt'
            '/home/vin/sequence-tagging/experiments/4/model=bert,model_type=bert-large-uncased,tokenizer_type=bert/ckpts_v0/checkpoints/last.ckpt'
        )
        #self.model = torch.quantization.quantize_dynamic(self.model,
        #                                                 {torch.nn.Linear},
        #                                                 dtype=torch.qint8)
        self.model = self.model.to(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def batching(self, sessionId: str, userIn: str,
                 prevTrnUserOut: Dict[str, List[str]]):
        # actually, one or more (userIn, sessionId) are in FIFO; len(userIn)
        # is greater than 0 and estimated to be less than
        # self.tokenizer.model_max_length
        # actually prevTrnUserOut comes from the session-state stored in Redis
        batch: Dict[str, Any] = self._bert_tokenize(
            [[sessionId, userIn, prevTrnUserOut]])
        if batch['error_msgs']:
            for sessionId, err_msg in zip(batch['dlgTrnId'],
                                          batch['error_msgs']):
                # for each sessionId, send them their error message; then
                # return;
                # only one entry in our case, so return now
                return err_msg

        # batch was tokenized without any errors
        bch_nnOut_userOut: List[Dict[str, List[str]]] = self._forward(batch)
        print(f"bch_nnOut_userOut= {bch_nnOut_userOut}\n")
        for sessionId, nnOut_userOut in zip(batch['dlgTrnId'],
                                            bch_nnOut_userOut):
            # for each sessionId, send them their nnOut_userOut; then return
            # only one entry in our case, so return now
            return nnOut_userOut

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
            if example[2]:
                bch_prevTrnUserOut.append(example[2])
            else:
                bch_prevTrnUserOut.append(Utilities.userOut_init())
            bch_history.append(
                Utilities.prevTrnUserOut2history(bch_prevTrnUserOut[-1]))

        # return_attention_mask must be True for the model to work properly
        bch_nnIn_tknIds = self.tokenizer(
            text=bch_history,
            text_pair=bch_userIn_filtered_wrds,
            is_split_into_words=True,
            padding=True,
            truncation='do_not_truncate',
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_overflowing_tokens=False).to(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # if truncation is needed, create error messages
        err_msgs = []
        if bch_nnIn_tknIds['input_ids'].shape[
                1] > self.tokenizer.model_max_length:
            bch_nnIn_tknIds_SEP_beginEnd: torch.Tensor = (
                bch_nnIn_tknIds['input_ids'] == 102).nonzero()
            for idx, bol in enumerate(
                (bch_nnIn_tknIds_SEP_beginEnd[1::2, 1] <=
                 self.tokenizer.model_max_length).tolist()):
                if bol:
                    err_msgs.append("Try again; your text is lost")
                else:
                    err_msgs.append("Your text is too long; send shorter text")

        for idx in range(len(examples)):
            map_tknIdx2wrdIdx.append(bch_nnIn_tknIds.word_ids(idx))

        return {
            'nnIn_tknIds': bch_nnIn_tknIds,
            'dlgTrnId': bch_dlgTrnId,
            'prevTrnUserOut': bch_prevTrnUserOut,
            'userIn_filtered_wrds': bch_userIn_filtered_wrds,
            'map_tknIdx2wrdIdx': map_tknIdx2wrdIdx,
            'error_msgs': err_msgs
        }

    def _forward(self, batch: Dict[str, Any]) -> List[Dict[str, List[str]]]:
        self.model.eval()
        with torch.no_grad():
            # following line has dependence on Lightning
            bch_nnOut_tknLblIds: torch.Tensor = self.model(
                batch['nnIn_tknIds'])  # model.forward()
        D_bch_nnIn_tknIds, D_bch_nnOut_tknLbls = self._convert_ids2tkns(
            batch['nnIn_tknIds']['input_ids'], bch_nnOut_tknLblIds)
        print(f"D_bch_nnIn_tknIds= {D_bch_nnIn_tknIds}")
        print(f"D_bch_nnOut_tknLbls= {D_bch_nnOut_tknLbls}")

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


'''
inference = Inference()
sessionId = 93
userIn = "1362287.04 dollars liytle 1992 genesis brand"
prevTrnUserOut = 0
nnOut_userOut = inference.batching(sessionId, userIn, prevTrnUserOut)
#assert nnOut_userOut == {'brand': ['genesis'], 'model': [], 'color': [], 'style': [], 'mileage': [], 'price': ['1362287.04 $'], 'year': ['less 1992']}

userIn = "smalker 1991 year frrightliner I want to buy fivian acentador"
prevTrnUserOut = nnOut_userOut
nnOut_userOut = inference.batching(sessionId, userIn, prevTrnUserOut)
#assert nnOut_userOut == {'brand': ['genesis', 'freightliner', 'rivian'], 'model': ['aventador'], 'color': [], 'style': [], 'mileage': [], 'price': ['1362287.04 $'], 'year': ['less 1992', 'less 1991']}
x = 1
'''
