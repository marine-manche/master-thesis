"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Selection of users in cities of interest
Determination of users' household location

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import pickle as pkl
import pandas as pd
import geopandas as gpd
import statistics as stat


def main():
    # Load activities and users
    os.system("git lfs pull -I 'data/aggregated-mobis-2/mobis_activities.csv'")
    activities = pd.read_csv("../data/aggregated-mobis-2/mobis_activities.csv")
    os.system("git lfs pull -I 'data/aggregated-mobis-2/mobis_tracked_participants.csv'")
    users = pd.read_csv("../data/aggregated-mobis-2/mobis_tracked_participants.csv")

    # Select relevant activities 
    activities_select = activities[activities['imputed_purpose'] == 'Home']

    # Compute median of x and y coordinates
    homes = users
    homes['med_x'] = 0
    homes['med_y'] = 0
    for i, user in users.iterrows():
        points = activities_select[activities_select['user_id'] == user['user_id']]
        if not points.empty:
            homes.loc[i, 'med_x'] = stat.median(points['geom_x'].values)
            homes.loc[i, 'med_y'] = stat.median(points['geom_y'].values)
    homes_gpd = gpd.GeoDataFrame(homes, geometry=gpd.points_from_xy(homes.med_x, homes.med_y), crs='EPSG:2056')

    # Select participants living in cities
    boundary = gpd.read_file('../data/geodata/swiss_cities_LV95.geojson')
    users_clip = homes_gpd[homes_gpd['geometry'].within(boundary['geometry'][0])]

    # Save DataFrames
    pkl.dump(homes_gpd, open("../pickles/users.pkl", "wb"))
    pkl.dump(users_clip, open("../pickles/users_cities.pkl", "wb"))


if __name__ == '__main__':
    main()
