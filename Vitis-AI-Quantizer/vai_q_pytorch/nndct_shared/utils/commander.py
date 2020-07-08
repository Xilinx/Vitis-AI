class BaseCommander:

  @classmethod
  def register(cls, obj, attr):
    if not hasattr(obj, attr):
      setattr(obj, attr, {})
    commander = cls()
    getattr(obj, attr).update(commander.get_all_commands())

  @classmethod
  def get_all_commands(cls):
    all_commands = {}
    commander = cls()
    while True:
      try:
        all_commands.update({k:v for k,v in commander.create_commands().items() if \
            k!='self' and not k.startswith('_') and not k in all_commands})
        commander = super(commander.__class__, commander)
      except AttributeError:
        break
    return all_commands
