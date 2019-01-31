import csv
import numpy as np

def markup():
    name = input("total set file name?\n")
    k = open("../../Training_Sets/" + name+ ".csv")
    rawset = list(csv.reader(k))
    try:
        m = open("../../Training_Sets/HOUR_" + name + ".csv", "w")
    except:
        print("please close the file!")
    writer = csv.writer(m, lineterminator="\n")
    copy = list()
    print(len(rawset))
    header = rawset[0]
    header[0] = "hours"
    del header[1:4]
    copy.append(rawset[0])

    MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthhours = [24*k for k in MONTHS]
    cumulativemonths = [sum(monthhours[0:k]) for k in range(12)]#this will get you the cumulative hours imparted by month
    print(cumulativemonths)

    for j in range(1, len(rawset)):

        hours = int(rawset[j][3]) + (int(rawset[j][2])-1)*24 + cumulativemonths[int(rawset[j][1])-1]

        carrier = rawset[j][4:19]
        carrier.insert(0, hours)
        copy.append(carrier)


    print(len(copy))
    writer.writerows(copy)
    return copy

def substitute(copy):


def main():
    pass



