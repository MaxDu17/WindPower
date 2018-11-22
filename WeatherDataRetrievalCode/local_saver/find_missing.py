import csv

name = input("total set file name?\n")
k = open("../../Training_Sets/" + name+ ".csv")
rawset = list(csv.reader(k))
m = open("../../Training_Sets/MARKED_" + name + ".csv", "w")
writer = csv.writer(m, lineterminator="\n")
offset = 0
print(len(rawset))
for j in range(1,366*24):
    try:
        if int(rawset[j][3]) != int((j-1-offset)%24):
            rawset.insert(j, ["DATA NOT FOUND"])
            #print(rawset[j][0:4])
    except:
        continue

print(len(rawset))
writer.writerows(rawset)
