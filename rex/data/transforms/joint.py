from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from rex.data.transforms.base import CachedTransformOneBase
from rex.data.collate_fn import GeneralCollateFn
import numpy as np
from rex.utils.progress_bar import pbar
import json
import torch
from rex.utils.io import load_json
import re

from collections import defaultdict
from typing import Any, Iterable, List, MutableSet, Optional, Tuple, TypeVar

Filled = TypeVar("Filled")

class MaxLengthExceedException(Exception):
    pass


class USMTransform(CachedTransformOneBase):
    def __init__(self, plm_dir: str, max_seq_len: int = 512) -> None:
        super().__init__()

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(plm_dir)
        self.lm_token = "[LM]"
        self.lp_token = "[LP]"
        self.text_token = "[T]"
        num_added = self.tokenizer.add_tokens([self.lm_token, self.lp_token, self.text_token], special_tokens=True)
        assert num_added == 3
        self.lm_token_id, self.lp_token_id, self.text_token_id = self.tokenizer.convert_tokens_to_ids([self.lm_token, self.lp_token, self.text_token])
        #编号为50000 50001 50002
        self.max_seq_len = max_seq_len


        self.collate_fn: GeneralCollateFn = GeneralCollateFn(
            {
                "input_ids": torch.long,
                "mask": torch.long,
                "ttl_labels": torch.long,
                "ltl_labels": torch.long,
                "tll_labels": torch.long,
                "label_map":torch.long,
            },
            guessing=False,
            missing_key_as_null=True,
        )
        self.collate_fn.update_before_tensorify = self.dynamic_padding



    def build_label_seq(self, ent_labels: set, relation_labels: set) -> tuple:
        """
        Returns:
            input_ids
            input_tokens
            mask
            label_map: {label index: {"type": "m"/"p", "string": "person"}, ...}
            label_str_to_idx: {("person", "m"): label index}
        """
        label_map = {}
        label_str_to_idx = {}
        mask = [1]
        input_tokens = [self.tokenizer.cls_token]
        for ent in ent_labels:
            input_tokens.append(self.lm_token)
            mask.append(2)
            label_index = len(input_tokens)#label_index就是第几个位置
            label_map[label_index] = {"type": "m", "string": ent}
            label_str_to_idx[(ent, "m")] = label_index
            label_tokens = self.tokenizer.tokenize(ent)#"人物简介"->['人', '物', '简', '介']
            input_tokens.extend(label_tokens)
            mask.extend([3] * len(label_tokens))
        for rel in relation_labels:
            input_tokens.append(self.lp_token)
            mask.append(4)
            label_index = len(input_tokens)
            label_map[label_index] = {"type": "p", "string": rel}
            label_str_to_idx[(rel, "p")] = label_index
            label_tokens = self.tokenizer.tokenize(rel)
            input_tokens.extend(label_tokens)
            mask.extend([5] * len(label_tokens))
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_ids, input_tokens, mask, label_map, label_str_to_idx

    def build_input_seq(self, tokens: list, ent_labels: set, rel_labels: set):#存放实体标签和关系的集合
        input_ids, input_tokens, mask, label_map, label_str_to_idx = self.build_label_seq(ent_labels, rel_labels)
        input_tokens.append(self.text_token)
        mask.append(6)
        offset = len(mask)#数据data之前的总长度
        remain_len = self.max_seq_len - offset - 1#最后还要补一个[SEP]
        if remain_len <= 0:
            raise MaxLengthExceedException
        
        # if remain_len>len(tokens):
        #     remain_tokens = tokens
        #     input_tokens.extend(remain_tokens)
        #     mask.extend([7] * len(tokens))
        #     input_tokens.append(self.tokenizer.sep_token)
        #     mask.append(8)
        #     l=len(mask)
        #     mask.extend([0]*self.max_seq_len-l)
       
        remain_tokens = tokens[:remain_len]#截断句子到最大句长
        input_tokens.extend(remain_tokens)
        mask.extend([7] * remain_len)
        input_tokens.append(self.tokenizer.sep_token)
        mask.append(8)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return input_ids, input_tokens, mask, label_map, label_str_to_idx, offset

    def transform(
            self, 
            instance: dict,
            debug: Optional[bool] = False,
            desc: Optional[str] = "Transform",
            disable_pbar: Optional[bool] = False,
            **kwargs) -> dict:
        """
        Args:
            instance: {
                "id": "idx".
                "tokens": ["token", "##1"],已经经过了一次tokenizer.tokenize(sentence)
                "ents": [[[start, end + 1], "label"], ...]
                "relations": [[[head start, head end + 1], "relation", [tail start, tail end + 1]], ...],
                "events": [{"event_type": "event type", "trigger": [start, end + 1], "arguments": [[[arg start, end + 1], "role"], ...]}, ...],
            }
        """
        ent_labels = set(x[1] for x in instance["ents"])
        # ent_labels.update(x["event_type"] for x in instance["events"])
        rel_labels = set(x[1] for x in instance["relations"])
        # rel_labels.update(x[1] for e in instance["events"] for x in e["arguments"])

        # ent_labels=set()
        # rel_labels=set()

        # for i in instance:
        #     if i["ents"]:
        #         for ent in i["ents"]:
        #             ent_labels.add(ent[1])
        #     if i["relations"]:
        #         for relation in i["relations"]:
        #             rel_labels.add(relation[1])

        tokens=self.tokenizer(instance["tokens"])
            
        input_ids, input_tokens, mask, label_map, label_str_to_idx, offset=self.build_input_seq(tokens,
                                                                                            ent_labels, rel_labels)
        
        ttl_labels=np.zeros((3,self.max_seq_len,self.max_seq_len),dtype= np.int64)
        ltl_labels=np.zeros((2,self.max_seq_len,self.max_seq_len),dtype= np.int64)
        tll_labels=np.zeros((2,self.max_seq_len,self.max_seq_len),dtype= np.int64)

        if instance["ents"]:
            for ent in instance["ents"]:
                ttl_labels[0,ent[0][0]+offset,ent[0][1]+offset]=1 #head,tail

                ltl_labels[0,label_str_to_idx[(ent[1],"m")],ent[0][0]+offset]=1 #label,head
                ltl_labels[1,label_str_to_idx[(ent[1],"m")],ent[0][1]+offset]=1 #1abel,tail

        if instance["relations"]:
            relation=instance["relations"]
            ttl_labels[0,relation[0][0]+offset,relation[0][1]+offset]=1 #head,tail
            ttl_labels[0,relation[2][0]+offset,relation[2][1]+offset]=1 #head,tail
            ttl_labels[1,relation[0][0]+offset,relation[2][0]+offset]=1 #head,head
            ttl_labels[2,relation[0][1]+offset,relation[2][1]+offset]=1 #tail,tail

            ltl_labels[0,label_str_to_idx[(relation[1],"p")],relation[2][0]+offset]=1 #label,head
            ltl_labels[1,label_str_to_idx[(relation[1],"p")],relation[2][1]+offset]=1 #1abel,tail

            tll_labels[0,relation[0][0]+offset,label_str_to_idx[(relation[1],"p")]]=1 #head,label
            tll_labels[1,relation[0][1]+offset,label_str_to_idx[(relation[1],"p")]]=1 #tail,label

        ins={
                "id": instance["id"],
                "input_ids": input_ids,
                "mask": mask,
                "ttl_labels": ttl_labels,
                "ltl_labels": ltl_labels,
                "tll_labels": tll_labels,
                "label_map" :label_map
            }
            
        return ins

    
    def dynamic_padding(self, data: dict) -> dict:
        data["input_ids"] = self.padding(data["input_ids"], self.tokenizer.pad_token_id)
        data["mask"] = self.padding(data["mask"], 0)
        return data

    def padding(self, batch_seqs: Iterable[Filled], fill: Filled) -> Iterable[Filled]:
        # max_len = max(len(seq) for seq in batch_seqs)
        # assert max_len <= self.max_seq_len
        max_len=self.max_seq_len
        for i in range(len(batch_seqs)):
            batch_seqs[i] = batch_seqs[i] + [fill] * (max_len - len(batch_seqs[i]))
        return batch_seqs
    
    
    def predict_transform(self,texts: List[str]) -> dict:
        result= []
        for text_id, text in enumerate(texts):
            data_id = f"Prediction#{text_id}"
            d=  {
                    "id": data_id,
                    "tokens": text,
                    "ents":[],
                    "relations":[],
                    "events":[]
                }
            final_data = self.transform(d, disable_pbar=True)
            result.append(final_data)
        return result

        
    
    


   
    
