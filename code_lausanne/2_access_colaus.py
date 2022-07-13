"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Compute share of nature for each nature type in surroundings of users' households

Dataset: CoLaus
Scale: Local

Author: Marine Manche
Date: 13/07/22
"""

import os
import pickle as pkl
import geopandas as gpd


def main():
    # Load files
    os.system("git lfs pull -I 'pickles/colaus/colaus.pkl'")
    users = pkl.load(open("../pickles/colaus/colaus.pkl", "rb"))
    users = users.to_crs('EPSG:2056')  # Change to CH1903+/LV95 crs

    os.system("git lfs pull -I 'data/geodata/land_use_geom_fixed.geojson'")
    lulc = gpd.read_file("../data/geodata/land_use_geom_fixed.geojson")
    buffer = gpd.read_file('../data/geodata/swiss_cities_buffer_single_parts.geojson')

    nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']

    # Assign cities to each user
    users['city'] = ''
    for i in range(len(buffer)):
        city = buffer.loc[i, 'AName']
        users.loc[users.intersects(buffer.loc[i, 'geometry']), 'city'] = city

        # Select users in Lausanne
    users = users[users['city'] == 'Lausanne']

    # Compute nature types area for each user
    for BUFFER in [400, 600, 800, 1000, 1200]:
        for var in nature + ['green']: users['exp_' + var + '_' + str(BUFFER)] = 0

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
                        users.loc[i, 'exp_' + var + '_' + str(BUFFER)] = temp / area
                    add += temp

                    # Get the total green share
            users.loc[i, 'exp_green_' + str(BUFFER)] = add / area

            # Compute distance to nearest green area
    for var in nature + ['green']: users['nearest_' + var] = 0

    count = 0
    for i, user in users.iterrows():
        count += 1
        print(count / len(users))
        for var in nature:
            lulc_temp = lulc[lulc['category'] == var]
            dists = []

            for j, polygon in lulc_temp.centroid.iteritems():
                dist = polygon.distance(users.loc[i, 'geometry'])
                dists.append(dist)
            users.loc[i, 'nearest_' + var] = min(dists)

        # Get the nearest green areas
        users.loc[i, 'nearest_green'] = min(users.loc[i, ['nearest_' + c for c in nature]])  # [m]

    # Save DataFrame
    pkl.dump(users, open("../pickles/colaus/colaus_exp.pkl", "wb"))


if __name__ == '__main__':
    main()
