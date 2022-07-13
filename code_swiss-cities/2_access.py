"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Compute share of nature for each nature type in surroundings of users' households

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import pickle as pkl
import geopandas as gpd

BUFFER = 600  # [m]


def main():
    # Load files
    os.system("git lfs pull -I 'pickles/users_cities.pkl'")
    users = pkl.load(open("../pickles/users_cities.pkl", "rb"))

    os.system("git lfs pull -I 'data/geodata/land_use_geom_fixed.geojson'")
    lulc = gpd.read_file("../data/geodata/land_use_geom_fixed.geojson")

    nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']

    # Compute nature share around address

    for var in nature + ['green']:
        users['exp_' + var] = 0

    # Compute nature types area for each user
    for i, user in users.iterrows():

        buff = user.geometry.buffer(BUFFER)
        user_cliped = gpd.clip(lulc, buff)

        area = buff.area
        add = 0

        if not user_cliped.empty:
            user_cliped['area'] = user_cliped.area
            tab = user_cliped.groupby(['category'])['area'].sum()

            for var in nature:
                temp = 0
                if var in tab.index:
                    temp = tab.loc[var]
                    users.loc[i, 'exp_' + var] = temp / area
                add += temp

        # Get the total green share
        users.loc[i, 'exp_green'] = add / area

    # Save DataFrame
    pkl.dump(users, open("../pickles/users_exp.pkl", "wb"))


if __name__ == '__main__':
    main()
