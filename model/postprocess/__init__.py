# ETESVS/postprocess/__init__.py
from model.registeries import POSTPROCESSING

from model.postprocess.ctc_like import CTCLabelDecode, DistillationCTCLabelDecode
from model.postprocess.transformer_like import NRTRLabelDecode

POSTPROCESSING.register(CTCLabelDecode)
POSTPROCESSING.register(DistillationCTCLabelDecode)

POSTPROCESSING.register(NRTRLabelDecode)
