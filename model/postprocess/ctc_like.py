import numpy as np
from model.postprocess.base import BaseRecLabelDecode
from model.postprocess.utils import tensor_to_numpy

class CTCLabelDecode(BaseRecLabelDecode):
    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        preds = tensor_to_numpy(preds)
        if label is not None:
            label = tensor_to_numpy(label)
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True, return_word_box=return_word_box)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        return ["blank"] + dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert
    Convert between text-label and text-index
    """

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        model_name=["student"],
        key=None,
        multi_head=False,
        **kwargs,
    ):
        super(DistillationCTCLabelDecode, self).__init__(
            character_dict_path, use_space_char
        )
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred["ctc"]
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output
    