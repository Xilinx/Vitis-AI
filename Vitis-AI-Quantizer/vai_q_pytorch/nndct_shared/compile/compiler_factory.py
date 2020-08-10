class CompilerFactory(object):

  @staticmethod
  def get_compiler(compile_backend):
    if compile_backend == 'xmodel':
      from .xir_compiler import XirCompiler
      return XirCompiler()
    else:
      raise NotImplementedError('other compiler is not implemented except xir')
