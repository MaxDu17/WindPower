import csv

name = input("total set file name?\n")
k = open("../../Training_Sets/" + name+ ".csv")
rawset = list(csv.reader(k))
m = open("../../Training_Sets/INTERPOLATED_" + name + ".csv", "w")
writer = csv.writer(m, lineterminator="\n")
copy = list()
print(len(rawset))
copy.append(rawset[0])
for j in range(1, len(rawset)-1):

    print(j)
    copy.append(rawset[j])
    #if int(rawset[j][3]) != int((j-1)%24):
    if int(rawset[j][3]) != (int(rawset[j+1][3]) - 1)%24:
        print("true")
        copy.append(["NO DATA FOUND"])
        #print(rawset[j][0:4])

print(len(copy))
writer.writerows(copy)
