import csv
import numpy as np
def main():
    file_directory = "../../Training_Sets/" + input("filename for weather?\n") + ".csv"
    weather_ = open(file_directory, "r")
    weather = list(csv.reader(weather_))

    file_directory = "../../Training_Sets/" + input("filename for wind power?\n") + ".csv"
    wind_ = open(file_directory, "r")
    wind = list(csv.reader(wind_))

    #this code assumes that the weather is read at the begining of the thing
    weather_counter = 1
    carrier = wind[0]
    carrier.extend(weather[0][1:])
    print(carrier)
    big_list = list()
    big_list.append(carrier)


    for i in range(1, len(wind)):
        c = wind[i]
        if int(wind[i][0]) == int(weather[weather_counter][0]) + 60:
            #print(wind[i][0])
            print(weather_counter)
            #input()
            weather_counter += 1
        c.extend(weather[weather_counter][1:])
        big_list.append(c)

    final_ = open("../../Training_Sets/ALL_DATA.csv", "w")
    final = csv.writer(final_, lineterminator = "\n")

    final.writerows(big_list)

    transposed_list = np.array(big_list[1:]).T.tolist()

    for parameter in transposed_list[1:]:
        print(parameter)
        maximum = max(parameter)
        maximum = float(maximum)
        minimum = min(parameter)
        minimum = float(minimum)
        range_ = maximum-minimum
        parameter = [(float(x) - minimum)/range_ for x in parameter]

    normalized_list = np.array(transposed_list).T.tolist()

    normalized_ = open("../../Training_Sets/ALL_DATA_NORMALIZED.csv", "w")
    normalized = csv.writer(normalized_, lineterminator = "\n")

    normalized.writerows(transposed_list)




if __name__ == "__main__":
    main()




