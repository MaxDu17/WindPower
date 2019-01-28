import csv
name = input("total set file name?\n")
k = open("../../Training_Sets/" + name+ ".csv")
rawset = list(csv.reader(k))
m = open("../../Training_Sets/INTERPOLATED_" + name + ".csv", "w")
writer = csv.writer(m, lineterminator="\n")
copy = list()
print(len(rawset))
copy.append(rawset[0])

MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for j in range(1, len(rawset)-1):
    hours = rawset[j][1] *

print(len(copy))
writer.writerows(copy)
