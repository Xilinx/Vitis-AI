import numpy as np

DiscROC = np.loadtxt('.DiscROC.txt')
for i in range(96, 109):
    index = np.where(DiscROC[:, 1] == i)[0]
    if index :
        break
print ( "Recall :" + str(np.mean(DiscROC[index], axis = 0)) )
