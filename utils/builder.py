class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def __contains__(self, key):
        return self._obj_map.get(key) is not None

    @property
    def name(self):
        return self._name

    def _do_register(self, name, obj):
        assert name not in self._obj_map, \
            f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        if obj is None:
            def deco(func_or_class, name=name):
                self._do_register(name or func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        self._do_register(name or obj.__name__, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret


def build(cfg, registry, key='name', **extra_kwargs):
    if cfg is None:
        return None
    if isinstance(cfg, list):
        return [build(c, registry, key=key, **extra_kwargs) for c in cfg]

    assert isinstance(cfg, dict) and key in cfg, \
        f"Config must be a dict with key '{key}', got: {cfg}"
    cfg_copy = cfg.copy()
    obj_type = cfg_copy.pop(key)

    cfg_copy.update(extra_kwargs)

    obj_cls = registry.get(obj_type)
    return obj_cls(**cfg_copy)
