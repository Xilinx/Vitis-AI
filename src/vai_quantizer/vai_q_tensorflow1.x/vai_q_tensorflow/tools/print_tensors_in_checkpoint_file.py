from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("./ckptdir/model.ckpt-2501",tensor_name='',\
        all_tensors=True)
