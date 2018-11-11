from ftplib import FTP
import csv
directory = "HAS011159558"
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
k = open("tarfiles.csv", "w")
tarfile_list = csv.writer(k, lineterminator ='\n')
ftp.cwd('pub/has/model/' + directory + '/')
try:
    content = ftp.nlst()
    print(len(content))
    for item in content:
        tarfile_list.writerow([item])
except:
    print("error, no files found. Quitting...")
    quit()