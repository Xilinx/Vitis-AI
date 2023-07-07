import numpy as np
import pickle
with open('./original.pickle', 'rb') as handle:
  original = pickle.load(handle)
with open('./folded.pickle', 'rb') as handle:
  folded = pickle.load(handle)

# import pdb; pdb.set_trace()
print(original.shape)
print(folded.shape)
print(np.max(np.abs(original-folded)))
