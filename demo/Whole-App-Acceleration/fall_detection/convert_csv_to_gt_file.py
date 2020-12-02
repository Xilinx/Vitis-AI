import pandas as pd

df = pd.read_csv('urfall-cam0-falls.csv', header=None, usecols=[0, 1, 2])

with open('gt.txt', "w") as f:
    for index, row in df.iterrows():
        file = "{}-cam0-rgb-{:03d}.png".format(row[0], row[1])
        label = int(row[2])
        if label != 0:
            label = 1
        f.write("{} {}\n".format(file, label))
