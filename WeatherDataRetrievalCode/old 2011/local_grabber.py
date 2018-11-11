from ftplib import FTP
import csv
import os
import time
import pygrib #library that only works in linux

hour = 5 #change me!
hour_big = "0" + str(hour) + ".g2/"
hour_sub = "_0" + str(hour) + "00_"

file_path = "/home/max/DRIVE/data/0" + str(hour) + "00/"
base_command = 'ruc2_130_'
year = '2011'


date_dict = {
1:"01", 2:"02", 3:"03",
4:"04", 5:"05", 6:"06",
7:"07", 8:"08", 9:"09",
10:"10", 11:"11", 12:"12",
13:"13", 14:"14", 15:"15",
16:"16", 17:"17", 18:"18",
19:"19", 20:"20", 21:"21",
23:"23", 24:"24", 25:"25",
26:"26", 27:"27", 28:"28",
29:"29", 30:"30", 31:"31",
22:"22"
} #how date is expressed

time_dict = {
    1: "001", 2: "002", 3: "003",
    4: "004", 5: "005", 6: "006",
    7: "007", 8: "008", 9: "009",
    10: "010", 11: "011", 12: "012",
    13: "013", 14: "014", 15: "015",
    16: "016", 17: "017", 18: "018", 0:"000",
} #how forcast time is expressed

category_dict = {0: "surface_pressure", 1: "temp@2M", 2: "wind_gust_speed", 3: "2_M_rel_humid", 4: "temp_gnd_lvl"}

keepers = [223,230,300,295,310] #the data points to keep

point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location

headers = ["year", "month", "date", "hour"]
error_headers = headers + ["forecast_iteration"]
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


try:
    k = open("2011_TOTALSET.csv", "r")
    print("existing file detected!")
    r = csv.reader(k) #this is crash protection to ensure that everything doesn't get erased
    lines = list(r)
    lines = lines[1:len(lines) + 1]
    for i in range(len(lines)): #this is a 'dumb' way of casting a list
        for j in range(len(lines[i])):
            lines[i][j] = float(lines[i][j])

    input("loaded previous data: " + str(len(lines)) + " lines of data. Press enter to continue")
    k.close()
    big_data_ = open("2011_TOTALSET.csv", "w")  # here we get the large file
    big_data = csv.writer(big_data_, lineterminator="\n")
    big_data.writerow(headers)
    lower_bound = lines.pop()
    big_data.writerows(lines)  # we write the headers here


except:
    input("no filled file detected. Starting from scratch. Press enter to continue")
    big_data_ = open("2011_TOTALSET.csv", "w") #here we get the large file
    big_data = csv.writer(big_data_, lineterminator = "\n")
    big_data.writerow() #we write the headers here
    lower_bound = [2011,1,1,hour]

error_file_ = open("error_file.csv", "w")
error_file = csv.writer(error_file_, lineterminator = "\n")
error_file.writerow(error_headers)

for l in range(int(lower_bound[1]),13):
    print("I'm on month: " + str(l))
    for j in range(int(lower_bound[2]),32):
        print("I'm on day: " + str(j))
        base_template = [2011, l, j, hour]
        for i in range(0,3):
            lower_bound = [2011,1,1,hour] #this is to prevent a logic error in which it skips to where it started diff month
            print("I'm on forecast hour " + str(i))
            address = file_path + base_command + year + date_dict[l]+date_dict[j] + hour_big +\
                base_command + year + date_dict[l]+date_dict[j] + hour_sub + time_dict[i] + ".grb2"

            try:
                opened_file = pygrib.open(address)
            except:
                if j == 31 or ((j == 28 or j == 29 or j == 30 ) and l == 2 ):
                    print("this month doesn't have a 31 (or 28/29 for feb!) Skipping...")
                    base_template = ["IGNORE"]
                else:
                    print("file not found, this is recorded in the database")
                    error_file.writerow([2011, l, j, hour, "forecast hour " + str(i)])
                    base_template.extend([-99999,-99999,-99999,-99999,-99999,]) #makes it robust to missing files
                continue

            delta_list = [k * delta[i] for k in gate_delta]
            ok_list = [sum(x) for x in zip(delta_list, keepers)]
            for number in ok_list:
                selection = opened_file.select()[number]
                print(selection)
                selection_ = selection.values
                single_pt = selection_[point_to_keep_i][point_to_keep_j]
                base_template.append(single_pt)
                print("extracted: " + str(number) + "\n")

        if base_template[0] != "IGNORE":
            big_data.writerow(base_template)


base_template = [2011,1,1,hour+1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
big_data.writerow(base_template) #this is done so the next itertion of time doesn't erase the past point.
