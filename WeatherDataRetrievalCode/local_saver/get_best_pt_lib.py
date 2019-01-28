"""
This program takes a set of coordinates and finds the best matching pair on a giant
grid of coordinates courtesy of the key given in the grib file. This will
be later used to grab a single value out of a a giant grib dataset in order to collect
the weather data needed for the location.
Target.csv is the coordinates that will be matched as closely as possible in the giant csv.
Results are saved in "best_coordinates_ruc2anl_130.csv"
"""
import csv
import pandas as pd

#i is rows, j is columns
class Searcher():
    def search(self):
        data_lat = pd.read_csv("../lats_ruc2_130.csv", header=None)
        data_lon = pd.read_csv("../lons_ruc2_130.csv",header=None)

        target_frame = pd.read_csv("../target.csv")
        lat_ = target_frame[["lat"]]
        long_ = target_frame[["long"]]

        lat = lat_.values
        long = long_.values
        lat = lat[0][0] #these are the extracted target values
        long = long[0][0]

        target = [lat,long]

        frame_lat = data_lat.values
        frame_lon = data_lon.values

        best_i =best_j = 0 #this is where the best value will be stored
        alphabet_dict = {'a': 1, 'c': 3, 'b': 2, 'e': 5, 'd': 4, 'g': 7,
                         'f': 6, 'i': 9, 'h': 8, 'k': 11, 'j': 10, 'm': 13,
                         'l': 12, 'o': 15, 'n': 14, 'q': 17, 'p': 16,
                         's': 19, 'r': 18, 'u': 21, 't': 20, 'w': 23,
                         'v': 22, 'y': 25, 'x': 24, 'z': 26}
        inverted_alphabet_dict = dict([v,k] for k,v in alphabet_dict.items())

        best_error = 99999999999999

        for i in range(len(frame_lat)):
            for j in range(len(frame_lat[0])):
                lat_error = (frame_lat[i][j]-lat)**2
                long_error = (frame_lon[i][j] - long)**2

                total_error = lat_error + long_error
                if total_error < best_error:
                    best_error = total_error
                    best_i = i
                    best_j = j


        return best_i, best_j

