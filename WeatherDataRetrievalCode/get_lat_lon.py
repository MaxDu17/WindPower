import pygrib
import csv
writerlat = open("lats_ruc2_130.csv", "w")
writer_objectlat = csv.writer(writerlat, lineterminator="\n")

writerlon = open("lons_ruc2_130.csv", "w")
writer_objectlon = csv.writer(writerlon, lineterminator="\n")
test_file = "ruc2_130_2011010100.g2/ruc2_130_20110101_0000_001.grb2"
opened_file = pygrib.open(test_file)
selection = opened_file.select()[0]
lats, lons = selection.latlons()
writer_objectlat.writerows(lats)
writer_objectlon.writerows(lons)
