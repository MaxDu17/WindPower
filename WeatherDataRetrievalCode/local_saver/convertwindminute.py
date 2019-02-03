import csv
def main():
    name = "104686-2010"
    k = open("../../Training_Sets/" + name + ".csv")
    rawset = list(csv.reader(k))
    try:
        m = open("../../Training_Sets/MINUTE_" + name + ".csv", "w")
    except:
        print("please close the file!")
        quit()
    writer = csv.writer(m, lineterminator="\n")
    copy = list()
    print(len(rawset))
    header = rawset[3]
    header[0] = "Minutes"
    del header[1:5]
    copy.append(header)

    MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthhours = [24 * k for k in MONTHS]
    cumulativemonths = [sum(monthhours[0:k]) for k in range(12)]  # this will get you the cumulative hours imparted by month
    cumulativeminutes = [60 * k for k in cumulativemonths]
    print(cumulativeminutes)

    for j in range(4, len(rawset)):
        minutes = int(rawset[j][4]) + int(rawset[j][3]) * 60 + (int(rawset[j][2]) - 1) * 24 * 60 + cumulativeminutes[int(rawset[j][1]) - 1]

        carrier = rawset[j][5:11]
        carrier.insert(0, minutes)

        copy.append(carrier)

    print(len(copy))
    writer.writerows(copy)

if __name__ == "__main__":
    main()