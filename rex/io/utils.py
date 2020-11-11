import re
import json
import pickle
from typing import Iterable, Any, Optional, List


def dump_json(obj, filepath, **kwargs):
    with open(filepath, 'wt', encoding='utf-8') as fout:
        json.dump(obj, fout, ensure_ascii=False, **kwargs)


def load_json(filepath, **kwargs):
    data = list()
    with open(filepath, 'rt', encoding='utf-8') as fin:
        data = json.load(fin, **kwargs)
    return data


def dump_line_json(obj, filepath, **kwargs):
    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = json.dumps(d, ensure_ascii=False, **kwargs)
            fout.write("{}\n".format(line_d))


def load_line_json(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


def dump_pickle(obj, filepath, **kwargs):
    with open(filepath, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


def load_pickle(filepath, **kwargs):
    data = []
    with open(filepath, 'rb') as fin:
        data = pickle.load(fin, **kwargs)
    return data


def dump_csv(obj: Iterable[Any], filepath: str, delimiter: Optional[str] = '\t'):
    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = delimiter.join(d)
            fout.write("{}\n".format(line_d))


def load_csv(
    filepath: str,
    title_row: bool, 
    title_keys: Optional[List[str]] = None,
    sep: Optional[str] = '\t'
) -> List:
    """load csv file

    Args:
        filepath: filepath to load
        title_row: has title in the first row or not?
                   If true, it'll return a list of dict where keys are from
                   the title, otherwise a list of str list.
        title_keys: if not `title_row`, you can set the title keys yourself.
        sep: separation char
    """
    data = list()
    title_keys = title_keys if title_keys else []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line_data = line.strip().split(sep)
            if title_row and idx == 0:
                title_keys = line_data
                continue
            if title_keys:
                if len(title_keys) != len(line_data):
                    raise RuntimeError(f"len of title keys: {title_keys}"
                            f" does not match the line data in line {idx + 1}"
                            f" in file: {filepath}")
                ins = {}
                for col, key in zip(line_data, title_keys):
                    ins[key] = col
            else:
                ins = line_data                    
            data.append(ins)
    return data
