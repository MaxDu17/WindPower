import csv


k = open("../2011_TOTALSET.csv")
rawset = list(csv.reader(k))
m = open("../MARKEDTOTALSET.csv", "w")
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
