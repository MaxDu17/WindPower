import csv

file_directory = "../../Training_Sets/" + input("filename?\n") + ".csv"
weather_ = open(file_directory, "r")
weather = list(csv.reader(weather_))

file_directory = "../../104686-2010.csv"
wind_ = open(file_directory, "r")
wind = list(csv.reader(weather_))

def markup(wind): #this turns the wind into minutes
    copy = list()
    MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthhours = [24*k for k in MONTHS]
    cumulativemonths = [sum(monthhours[0:k]) for k in range(12)]#this will get you the cumulative hours imparted by month
    print(cumulativemonths)

    for j in range(1, len(wind)):

        hours = int(wind[j][3]) + (int(wind[j][2])-1)*24 + cumulativemonths[int(wind[j][1])-1]

        carrier = wind[j][4:19]
        carrier.insert(0, hours)

        copy.append(carrier)

    print(len(copy))
    writer.writerows(copy)
    return copy

