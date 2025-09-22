from utils.builder import Registry, build

DATASET = Registry('dataset')



def build_dataset(cfg):
    """Build dataset."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in DATASET:
        return build(cfg, DATASET)
    raise ValueError(f'{obj_type} is not registered in '
                     'DATASET')