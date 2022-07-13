"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Extract waypoints of users of interest

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import pickle as pkl
import pandas as pd
import os
import bz2
import sys

CHUNK = 16 * 1024


def main():
    c = sys.argv[1]

    users = pkl.load(open("../pickles/users_cities.pkl", "rb"))
    users = users.user_id.unique()

    os.system("git lfs pull -I 'data/mobis-waypoints-z/waypoints/" + c + ".csv.bz2'")

    # Decompress zip file and store in csv file
    filepath = "../csv/waypoints_pre/"
    waypoints_data_folder = "../data/mobis-waypoints-z/waypoints/"
    req = open(waypoints_data_folder + c + ".csv.bz2", 'rb')

    decompressor = bz2.BZ2Decompressor()
    with open(filepath + "waypoints_" + c + ".csv", 'wb') as fp:
        while True:
            chunk = req.read(CHUNK)
            if not chunk:
                break
            fp.write(decompressor.decompress(chunk))
    req.close()

    start = True
    filepath_out = "../csv/waypoints_post/"
    for waypoints in pd.read_csv(filepath + "waypoints_" + c + ".csv", chunksize=10000):
        waypoints_select = waypoints[waypoints.user_id.isin(users)]
        if start:
            start = False
            waypoints_select.to_csv(filepath_out + "waypoints_" + c + ".csv", index=False)
        else:
            waypoints_select.to_csv(filepath_out + "waypoints_" + c + ".csv", mode='a', index=False, header=False)

    waypoints_csv = pd.read_csv(filepath_out + "waypoints_" + c + ".csv")

    # Save DataFrame
    pkl.dump(waypoints_csv, open("../pickles/waypoints/waypoints_" + c + ".pkl", "wb"))


if __name__ == '__main__':
    main()
