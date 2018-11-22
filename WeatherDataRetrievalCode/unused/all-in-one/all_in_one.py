#!/usr/bin/env python3
import csv
import subprocess, os
from ftplib import FTP
#path = "/home/max/DRIVE/data/"
path = '/home/set/Max/data/'
len_path = len(path)
directory = "HAS011159558"
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
ftp.cwd('pub/has/model/' + directory + '/')
try:
    content = ftp.nlst()
except:
    print("error, no files found. Quitting...")
    quit()


print("attempting crash recovery")
crash_list = os.listdir(path + 'crash/')
k_ = open("tarfiles.csv", "r")
big_tar_list = list(csv.reader(k_))
big_tar_list = [m[0] for m in big_tar_list]
k_.close()

for item in crash_list:
    print(item)
    big_tar_list.remove(item) #now we are left with a list of things to do

input("crash recovery complete. " + str(len(big_tar_list)) + " files to go!")
big_tar_list = sorted(big_tar_list, key = lambda file_names: int(file_names[13:17]))
for filename in big_tar_list:
    # Download the file from the FTP server
    command = 'RETR ' + filename
    print("Downloading: " + filename)
    overarching_name = path + filename
    ftp.retrbinary(command, open(overarching_name, 'wb').write)

    # Untar each file to its own folder, after it is done, delete the tar file
    dirname = overarching_name.replace('.tar','')
    tarcommand = 'tar -xf '+overarching_name + ' -C ' + dirname + '; rm '+overarching_name + \
                 ';  ~/miniconda3/bin/python3 local_grabber_single.py ' + dirname + " " + filename
    print("Extracting tar: " + filename)
    subprocess.call(["mkdir", "-p", dirname])
    subprocess.Popen(['/bin/sh', '-c', tarcommand])
