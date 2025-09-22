import numpy as np
from model.postprocess.base import BaseRecLabelDecode
from model.postprocess.utils import tensor_to_numpy

class NRTRLabelDecode(BaseRecLabelDecode):
    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)) and len(preds) == 2:
            preds_id, preds_prob = tensor_to_numpy(preds[0]), tensor_to_numpy(preds[1])
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]; preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None: return text
            label = self.decode(tensor_to_numpy(label)[:, 1:])
            return text, label
        preds = tensor_to_numpy(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None: return text
        label = self.decode(tensor_to_numpy(label)[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        return ["blank", "<unk>", "<s>", "</s>"] + dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        batch_size = len(text_index)
        for b in range(batch_size):
            char_list, conf_list = [], []
            for i in range(len(text_index[b])):
                try:
                    ch = self.character[int(text_index[b][i])]
                except Exception:
                    continue
                if ch == "</s>":
                    break
                char_list.append(ch)
                conf_list.append(text_prob[b][i] if text_prob is not None else 1)
            result_list.append(("".join(char_list), np.mean(conf_list).tolist()))
        return result_list
