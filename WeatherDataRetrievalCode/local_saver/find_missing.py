import csv
import numpy as np
name = input("total set file name?\n")
k = open("../../Training_Sets/" + name+ ".csv")
rawset = list(csv.reader(k))
m = open("../../Training_Sets/TEST_" + name + ".csv", "w")
writer = csv.writer(m, lineterminator="\n")
copy = list()
print(len(rawset))
copy.append(rawset[0])

MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monthhours = [24*k for k in MONTHS]
cumulativemonths = [sum(monthhours[0:k]) for k in range(12)]#this will get you the cumulative hours imparted by month
print(cumulativemonths)

for j in range(1, len(rawset)-1):

    hours = int(rawset[j][3]) + int(rawset[j][2])*24 + cumulativemonths[int(rawset[j][1])-1]

    copy.append([hours, rawset[j][3:]])

print(len(copy))
writer.writerows(copy)
