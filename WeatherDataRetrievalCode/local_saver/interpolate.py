import csv
import numpy as np


def markup(name):

    k = open("../../Training_Sets/" + name+ ".csv")
    rawset = list(csv.reader(k))
    try:
        m = open("../../Training_Sets/HOUR_" + name + ".csv", "w")
    except:
        print("please close the file!")
        quit()
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
    header = copy[0]
    del copy[0]
    new_list = [0] * 8760
    for k in copy:
        new_list[int(k[0])] = k[1:] #creates a sparse array

    print(new_list[7])
    for i in range(len(new_list)):
        range_counter = 1

        if(not(new_list[i])):
            while not(new_list[i+range_counter]): #checks if blank is isolated
                range_counter +=1

            new_list[i] = new_list[i+range_counter]
        final = list()

    for i in range(8760):
        carrier = list(new_list[i])
        carrier.insert(0, i)
        final.append(carrier)

    final.insert(0, header)

    return final





def main():
    name = input("total set file name?\n")
    large_array = markup(name)
    final_result = substitute(large_array)

    m = open("../../Training_Sets/INTERPOLATED_" + name + ".csv", "w")

    writer = csv.writer(m, lineterminator="\n")
    writer.writerows(final_result)



if __name__ == "__main__":
    main()
    print("all  done!")

