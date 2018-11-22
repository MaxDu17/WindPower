#this program takes a .csv and normalizes down the y axis, or through the entire time series.
#used for current weather data
import pandas as pd
import numpy as np
import csv

def min_max(clean):
    minimum = np.amin(clean)
    maximum = np.amax(clean)
    range = maximum-minimum
    scaled_array = clean - minimum
    scaled_array = scaled_array / range
    return scaled_array

name = input("file name?\n")
data = pd.read_csv("../Training_Sets/" + name + ".csv", skiprows=3)  # read file

power_native = data[["power (MW)"]].values.tolist()
power = min_max(data[["power (MW)"]].values).tolist()
wind_dir = min_max(data[["wind direction at 100m (deg)"]].values).tolist()
wind_speed = min_max(data[["wind speed at 100m (m/s)"]].values).tolist()
air_temp = min_max(data[["air temperature at 2m (K)"]].values).tolist()
SAP = min_max(data[["surface air pressure (Pa)"]].values).tolist()
air_density = min_max(data[["density at hub height (kg/m^3)"]].values).tolist()

logger_ = open("../Training_Sets/" + name + "_NORMALIZED.csv", "w")
logger = csv.writer(logger_, lineterminator = '\n')

headers = ["native_power", "power", "wind_dir", "wind_speed", "air_temp", "SAP", "air_density"]
logger.writerow(headers)
for i in range(len(power)):
    k = [power_native[i][0], power[i][0], wind_dir[i][0], wind_speed[i][0], air_temp[i][0], SAP[i][0], air_density[i][0]]
    logger.writerow(k)