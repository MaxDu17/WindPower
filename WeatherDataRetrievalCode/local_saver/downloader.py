#!/usr/bin/env python3
import csv
import subprocess, os
from ftplib import FTP
#path = "/home/max/DRIVE/data/"
path = '/home/wedu/database/data/'
len_path = len(path)

k_ = open("tarfiles.csv", "r")
big_tar_list = list(csv.reader(k_))
big_tar_list = [m[0] for m in big_tar_list]
directory = big_tar_list.pop(0) #getting directory from tarfile list
k_.close()

#directory = input("what is the directory name?\n")
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

for item in crash_list:
    try:
        print(item)
        big_tar_list.remove(item) #now we are left with a list of things to do
    except:
        print("you probably forgot to clear the crash directory. Do that please!")
        quit()

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
        '; cd ' + path + 'crash/;'+ ' touch ' + filename
    print("Extracting tar: " + filename)
    subprocess.call(["mkdir", "-p", dirname])
    subprocess.Popen(['/bin/sh', '-c', tarcommand])
    print("done with " + filename + ". Added its name to the crash file")
