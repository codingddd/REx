import copy
from multiprocessing import Pool
from collections import defaultdict
from typing import Iterable, Optional, Any, List

from loguru import logger

from rex.data.vocab import Vocab
from rex.data.label_encoder import LabelEncoder
from rex.utils.position import construct_relative_positions, find_all_positions
from rex.utils.mask import construct_piecewise_mask
from rex.utils.progress_bar import tqdm


class TransformBase(object):
    def __init__(self, max_seq_len) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab = Vocab()

    def transform(self, lines: Iterable):
        raise NotImplementedError

    def predict_transform(self, strings: Iterable[str]):
        ret_data = []
        for string in strings:
            seq_len = min(len(string), self.max_seq_len)
            comp_len = max(0, (self.max_seq_len - seq_len))
            ret_data.append({
                "token_ids": self.vocab.convert_tokens_to_ids(list(string))[:seq_len] + comp_len * [self.vocab.pad_idx],
                "mask": seq_len * [1] + comp_len * [0],
            })
        return ret_data

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)


class CachedMCMLSentRETransform(TransformBase):
    """Data transform for cached multi-class multi-label sentence-level relation extraction task."""
    def __init__(self, max_seq_len) -> None:
        super().__init__(max_seq_len)
        self.label_encoder = LabelEncoder()

    def transform(self, data: Iterable, desc: Optional[str] = "Transform",
                  debug: Optional[bool] = False) -> List[dict]:
        num_truncated_rels = 0
        final_data = []
        if debug:
            data = data[:5]
        transform_loader = tqdm(data, desc=desc)

        for d in transform_loader:
            ent_validation = []
            for ent in d['entities']:
                if ent[2] > self.max_seq_len:
                    ent_validation.append(False)
                    continue
                ent_validation.append(True)
            valid_ent_pair2rels = defaultdict(set)
            for rel in d['relations']:
                if ent_validation[rel[1]] is False or ent_validation[rel[2]] is False:
                    num_truncated_rels += 1
                    continue
                valid_ent_pair2rels[(rel[1], rel[2])].add(self.label_encoder.update_encode_one(rel[0]))
            if len(valid_ent_pair2rels) == 0:
                continue
            token_ids, _ = self.vocab.encode(d['tokens'], self.max_seq_len, update=True)
            for ent_pair, rels in valid_ent_pair2rels.items():
                head_pos = construct_relative_positions(d['entities'][ent_pair[0]][1], self.max_seq_len)
                tail_pos = construct_relative_positions(d['entities'][ent_pair[1]][1], self.max_seq_len)
                final_data.append({
                    "id": d['id'],
                    "token_ids": token_ids,
                    "mask": construct_piecewise_mask(
                        d['entities'][ent_pair[0]][1],
                        d['entities'][ent_pair[1]][1],
                        min(len(d['tokens']), self.max_seq_len),
                        self.max_seq_len),
                    "labels": rels,
                    "head_pos": head_pos,
                    "tail_pos": tail_pos
                })
        final_data = list(filter(lambda x: x is not None, final_data))
        for d in final_data:
            d['labels'] = self.label_encoder.to_binary_labels(d['labels'])
        logger.info(transform_loader)
        logger.warning(f"#truncated_rels: {num_truncated_rels}")
        return final_data

    def predict_transform(self, obj: dict):
        """
        Args:
            obj:
                {
                    "text": "text",
                    "head": "head word",
                    "tail": "tail word"
                }
        """
        if obj['head'] not in obj['text'] or obj['tail'] not in obj['text']:
            raise ValueError(f"{obj['head']} or {obj['tail']} is not in {obj['text']}")
        head_pos = find_all_positions(obj['text'], obj['head'])[0]
        tail_pos = find_all_positions(obj['text'], obj['tail'])[0]
        if head_pos[1] > self.max_seq_len:
            logger.warning("head entity truncated")
            head_pos = [self.max_seq_len - 1, self.max_seq_len]
        if tail_pos[1] > self.max_seq_len:
            logger.warning("tail entity truncated")
            tail_pos = [self.max_seq_len - 1, self.max_seq_len]
        token_ids, _ = self.vocab.encode(list(obj['text']), self.max_seq_len, update=False)
        d = {
            "id": "",
            "token_ids": token_ids,
            "mask": construct_piecewise_mask(
                head_pos[0], tail_pos[0],
                min(len(obj['text']), self.max_seq_len),
                self.max_seq_len),
            "labels": None,
            "head_pos": construct_relative_positions(head_pos[0], self.max_seq_len),
            "tail_pos": construct_relative_positions(tail_pos[0], self.max_seq_len)
        }
        return d


class CachedMCBagRETransform(TransformBase):
    """Data transform for cached bag-level relation classification task."""
    def __init__(self, max_seq_len) -> None:
        super().__init__(max_seq_len)
        self.label_encoder = LabelEncoder()

    def transform(self, data: Iterable, desc: Optional[str] = "Transform",
                  debug: Optional[bool] = False) -> List[dict]:
        num_truncated_rels = 0
        final_data = []
        bag_scope = []
        bag2id = {}
        bag2label = {}

        if debug:
            data = data[:5]
        transform_loader = tqdm(data, desc=desc)

        for d in transform_loader:
            ent_validation = []
            for ent in d['entities']:
                if ent[2] > self.max_seq_len:
                    ent_validation.append(False)
                    continue
                ent_validation.append(True)
            valid_ent_pair2rels = defaultdict(set)
            for rel in d['relations']:
                if ent_validation[rel[1]] is False or ent_validation[rel[2]] is False:
                    num_truncated_rels += 1
                    continue
                valid_ent_pair2rels[(rel[1], rel[2])].add(self.label_encoder.update_encode_one(rel[0]))
            if len(valid_ent_pair2rels) == 0:
                continue
            token_ids, _ = self.vocab.encode(d['tokens'], self.max_seq_len, update=True)
            for ent_pair, rels in valid_ent_pair2rels.items():
                head = "_".join(d['tokens'][d['entities'][ent_pair[0]][1]:d['entities'][ent_pair[0]][2]])
                tail = "_".join(d['tokens'][d['entities'][ent_pair[1]][1]:d['entities'][ent_pair[1]][2]])
                bag_pair = f"{head}###{tail}"
                if bag_pair not in bag2id:
                    bag2id[bag_pair] = len(bag2id)
                    bag_scope.append([])
                    bag2label[bag_pair] = rels.pop()
                bag_scope[bag2id[bag_pair]].append(len(final_data))
                head_pos = construct_relative_positions(d['entities'][ent_pair[0]][1], self.max_seq_len)
                tail_pos = construct_relative_positions(d['entities'][ent_pair[1]][1], self.max_seq_len)
                final_data.append({
                    "id": d['id'],
                    "token_ids": token_ids,
                    "mask": construct_piecewise_mask(
                        d['entities'][ent_pair[0]][1],
                        d['entities'][ent_pair[1]][1],
                        min(len(d['tokens']), self.max_seq_len),
                        self.max_seq_len),
                    # MC setting, only take one relation
                    "labels": bag2label[bag_pair],
                    "head_pos": head_pos,
                    "tail_pos": tail_pos
                })
        logger.info(transform_loader)
        logger.warning(f"#truncated_rels: {num_truncated_rels}")
        return final_data, bag_scope

    def predict_transform(self, obj: dict):
        """
        Args:
            obj:
                {
                    "text": "text",
                    "head": "head word",
                    "tail": "tail word"
                }
        """
        if obj['head'] not in obj['text'] or obj['tail'] not in obj['text']:
            raise ValueError(f"{obj['head']} or {obj['tail']} is not in {obj['text']}")
        obj['text'] = obj['text'].split()
        head_pos = find_all_positions(obj['text'], obj['head'])[0]
        tail_pos = find_all_positions(obj['text'], obj['tail'])[0]
        if head_pos[1] > self.max_seq_len:
            logger.warning("head entity truncated")
            head_pos = [self.max_seq_len - 1, self.max_seq_len]
        if tail_pos[1] > self.max_seq_len:
            logger.warning("tail entity truncated")
            tail_pos = [self.max_seq_len - 1, self.max_seq_len]
        token_ids, _ = self.vocab.encode(list(obj['text']), self.max_seq_len, update=False)
        d = {
            "id": "",
            "token_ids": token_ids,
            "mask": construct_piecewise_mask(
                head_pos[0], tail_pos[0],
                min(len(obj['text']), self.max_seq_len),
                self.max_seq_len),
            "labels": None,
            "head_pos": construct_relative_positions(head_pos[0], self.max_seq_len),
            "tail_pos": construct_relative_positions(tail_pos[0], self.max_seq_len)
        }
        return d, [[0]]
