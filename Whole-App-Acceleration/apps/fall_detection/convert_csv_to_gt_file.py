import csv

with open('gt.txt', "w") as f:
    with open('urfall-cam0-falls.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            file = "{}-cam0-rgb-{:03d}.png".format(row[0], int(row[1]))
            label = int(row[2])
            if label != 0:
                label = 1
            f.write("{} {}\n".format(file, label))
