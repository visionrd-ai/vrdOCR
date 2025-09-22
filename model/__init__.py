from model.architecture import (vrdOCR)
from model.head import (CTCHead, NRTRHead, MultiHead)
from model.neck import (
    register_default_necks
)
from model.postprocess import (CTCLabelDecode, DistillationCTCLabelDecode, NRTRLabelDecode)
from model.backbone import (register_hrnet_variants)


# register_default_necks()
# register_hrnet_variants()   

