from ftplib import FTP
import csv
import subprocess
directory = input("what is the directory name?\n")
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
k = open("tarfiles.csv", "w")
tarfile_list = csv.writer(k, lineterminator ='\n')
ftp.cwd('pub/has/model/' + directory + '/')
try:
    content = ftp.nlst()
    print("I found " + str(len(content)) + " files!")
    tarfile_list.writerow([directory])
    for item in content:
        tarfile_list.writerow([item])
    answer = input("Would you like to delete crash files? (y/n)")
    if answer == 'y':
        print("deleting crash directory")
        subprocess.call(['rm','-r','/home/wedu/database/data/crash/'])
        subprocess.call(["mkdir", '/home/wedu/database/data/crash/'])
except:
    print("error, no files found. Quitting...")
    quit()