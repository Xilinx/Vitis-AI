import subprocess
import os

class CaffeFrontend():
  
  def __init__(self,**kwargs):
    arglist = []
    #here = os.path.dirname(os.path.realpath(__file__))  
    path = os.getenv("CONDA_PREFIX") + "/bin/decent_q"
    arglist.append(path)
    arglist.append("quantize")
    for k,v in kwargs.items():
      if k in ["caffeRoot","auto_test"]:
        continue
      arglist.append("-"+str(k))
      arglist.append(str(v))
    if "auto_test" in kwargs:
      if kwargs["auto_test"]:
        arglist.append("-auto_test")
    self.args = arglist
  
  def quantize(self):
    #print self.args
    os.environ["DECENT_DEBUG"] = "1"
    subprocess.call(self.args)
    
