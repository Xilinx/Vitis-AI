import numpy as np
import pickle
with open('./test/org_w.pickle', 'rb') as handle:
  org_w = pickle.load(handle)
with open('./test/org_b.pickle', 'rb') as handle:
  org_b = pickle.load(handle)
with open('./test/offset.pickle', 'rb') as handle:
  offset = pickle.load(handle)
with open('./test/scale.pickle', 'rb') as handle:
  scale = pickle.load(handle)


with open('./test/fold_w.pickle', 'rb') as handle:
  fold_w = pickle.load(handle)
with open('./test/fold_b.pickle', 'rb') as handle:
  fold_b = pickle.load(handle)

new_w = np.zeros_like(org_w)
new_b = np.zeros_like(org_b)
h, w, out, in_num = new_w.shape
for i in range(out):
  for l in range(in_num):
    for j in range(h):
      for k in range(w):
        new_w[j,k,i,l] = org_w[j,k,i,l] * scale[i]
new_b = scale * org_b + offset

print(np.max(np.abs(new_w-fold_w)))
print(np.max(np.abs(new_b-fold_b)))
import pdb; pdb.set_trace()
print(new_w.shape)
