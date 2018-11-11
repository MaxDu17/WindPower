import csv
import os
import pygrib #library that only works in linux
from sys import argv
import subprocess

script, dir_name, filename = argv
#path = "/home/max/DRIVE/data/crash/"
path = '/home/set/Max/data/crash/'
category_dict = {0: "surface_pressure", 1: "temp@2M", 2: "wind_gust_speed", 3: "2_M_rel_humid", 4: "temp_gnd_lvl"}

keepers = [223,230,300,295,310] #the data points to keep
headers = ["year", "month", "date", "hour"]
point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location

#delta = [0,0,2,2,0,2,2,0,2,2,0,2,2,0,2,2,0,2,2] #this compensates for the index-hopping that the dataset does
delta = [0,0,2]
gate_delta = [0,0,1,1,1]
lower_bound = list()

for i in range(3):
    for j in range(5):
        time = "forecast " + str(i) + "-"
        category = category_dict[j]
        concat = time + category
        headers.append(concat)

print("saving csv to: " + dir_name)
big_data_ = open(dir_name + "_data.csv", "w") #here we get the large file
big_data = csv.writer(big_data_, lineterminator = "\n")
big_data.writerow(headers)



file_names = os.listdir(dir_name)
i = 0

file_data = file_names[0] #this is to build up the template
year = file_data[9:13]
month = file_data[13:15]
date = file_data[15:17]
hour = file_data[18:20]
base_template = [year, month, date, hour]

file_names = sorted(file_names, key = lambda file_names: int(file_names[-7:-5]))
for file_name in file_names[0:3]:
    #ruc2_130_20110102_0500_004
    opened_file = pygrib.open(dir_name + "/" + file_name)

    delta_list = [k * delta[i] for k in gate_delta]
    ok_list = [sum(x) for x in zip(delta_list, keepers)]
    for number in ok_list:
        selection = opened_file.select()[number]
        #print(selection)
        selection_ = selection.values
        single_pt = selection_[point_to_keep_i][point_to_keep_j]
        base_template.append(single_pt)
        print("extracted: " + str(number) + " from " + file_name)
    i += 1
    opened_file.close()
big_data.writerow(base_template)
big_data_.close()
subprocess.call(["rm", "-r", dir_name])
k = open(path + filename, "w")
print("done with " + filename + ". Added its name to crash file")
k.close()