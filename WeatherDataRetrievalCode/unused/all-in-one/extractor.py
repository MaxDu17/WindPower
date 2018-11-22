import os
import csv

path = "../data-23/"

csv_list = [fname for fname in os.listdir(path) if fname.endswith('.csv')]


sorted_list = sorted(csv_list, key = lambda file_names: int(file_names[13:17]))
m_ = open("../2011_TOTALSET.csv", "r")
m = list(csv.reader(m_))
m_.close()
k_ = open("../2011_TOTALSET.csv", "w")
k = csv.writer(k_, lineterminator = '\n')

k.writerows(m)

for file in sorted_list:
    print(file)
    w = open(path + file, "r")
    opened_file = list(csv.reader(w))[1]
    k.writerow(opened_file)

k_.close()