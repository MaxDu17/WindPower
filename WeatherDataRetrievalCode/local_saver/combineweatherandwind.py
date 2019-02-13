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
    big_list = list()
    big_list.append(carrier)


    for i in range(1, len(wind)):
        c = wind[i]
        if int(wind[i][0]) == int(weather[weather_counter][0]) + 60:
            weather_counter += 1
        c.extend(weather[weather_counter][1:])
        big_list.append(c)

    final_ = open("../../Training_Sets/ALL_DATA.csv", "w")
    final = csv.writer(final_, lineterminator = "\n")

    final.writerows(big_list)

    headers = big_list[0]
    transposed_list = np.array(big_list[1:]).T.tolist()
    print(headers)

    normalized_list = list()
    normalized_list.append(transposed_list[0:1])
    print(len(normalized_list))

    for parameter in transposed_list[2:]:
        print(parameter[0])
        for k in range(0, len(parameter)): #this casts each from string to float
            try:
                parameter[k] = float(parameter[k])
            except:
                quit()
        maximum = max(parameter)
        minimum = min(parameter)
        range_ = maximum-minimum
        normalized_list.append([(x - minimum)/range_ for x in parameter])

    print()

    print(len(normalized_list))
    normalized_list = np.array(normalized_list).T.tolist()


    normalized_list.insert(0, headers)
    normalized_ = open("../../Training_Sets/ALL_DATA_NORMALIZED.csv", "w")
    normalized = csv.writer(normalized_, lineterminator = "\n")

    normalized.writerows(normalized_list)




if __name__ == "__main__":
    main()




