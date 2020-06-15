
def get_kwargs_or_defaults(kwargs, default_attrs, keys=None):
  keys = keys or default_attrs.keys()
  attr_dict = {}
  for k in keys:
    if isinstance(default_attrs[k], dict):
      attr_dict[k] = default_attrs[k]
      if k in kwargs and isinstance(kwargs[k], dict):
        attr_dict[k].update(kwargs[k])
    else:
      try:
        attr_dict[k] = kwargs[k]
      except:
        attr_dict[k] = default_attrs[k]
  return attr_dict

def set_kwargs_or_defaults(obj, kwargs, default_attrs=None, keys=None):
  if default_attrs:
    keys = keys or default_attrs.keys()
    attrs_dict = get_kwargs_or_defaults(kwargs, default_attrs, keys)
  else:
    keys = keys or kwargs.keys()
    attrs_dict = kwargs
  for k, v in attrs_dict.items():
    setattr(obj, k, v)
    
def nndct_pre_processing(init_func):

  def wrapper(obj, *args, **kwargs):
    if hasattr(obj, 'default_kwargs'):
      set_kwargs_or_defaults(obj, kwargs, obj.default_kwargs)
    if hasattr(obj, 'indexed_title'):
      assert isinstance(obj.indexed_title, list)
      for key in obj.indexed_title:
        setattr(obj, 'POS_' + key.upper(), obj.indexed_title.index(key))
    if hasattr(obj, 'default_commanders'):
      for k, v in obj.default_commanders.items():
        k.register(obj, v)
    init_func(obj, *args, **kwargs)

  return wrapper

def not_implement(func):

  def wrapper(obj, *args, **kwargs):
    func(obj, *args, **kwargs)
    raise NotImplemented("{} {}".format(obj, func.__name__))

  return wrapper
