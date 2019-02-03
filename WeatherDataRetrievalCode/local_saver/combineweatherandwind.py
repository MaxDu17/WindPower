import csv

file_directory = "../../Training_Sets/" + input("filename?\n") + ".csv"
weather_ = open(file_directory, "r")
weather = list(csv.reader(weather_))

file_directory = "../../104686-2010.csv"
wind_ = open(file_directory, "r")
wind = list(csv.reader(weather_))

