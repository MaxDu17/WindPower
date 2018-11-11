import pygrib
import csv

writer = open("data.csv", "w")
recorder = csv.writer(writer, lineterminator="\n")
test_file = "ruc2_130_2011010100.g2/ruc2_130_20110101_0000_001.grb2"
opened_file = pygrib.open(test_file)
selection = opened_file.read()
selection = list(selection)

for single in selection:
    recorder.writerow([single])
