"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Remove waypoints with low accuracy
Crop trips to cities' buffers
Split trips whose distance between point coordinates is too high

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import pandas as pd
import os
import pickle as pkl
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

ACCURACY = 100
DISTANCE = 1000  # [m]
c = 'B'


def main():
    # Load cities buffer
    os.system("git lfs pull -I 'data/geodata/swiss_cities_buffer_single_parts.geojson'")
    buffer = gpd.read_file('../data/geodata/swiss_cities_buffer_single_parts.geojson')

    # Load data and create Geodataframe
    os.system("git lfs pull -I 'pickles/waypoints_smoothed/waypoints_" + c + ".pkl'")
    waypoints = pkl.load(open("../pickles/waypoints_smoothed/waypoints_" + c + ".pkl", "rb"))

    waypoints = gpd.GeoDataFrame(waypoints, geometry=gpd.points_from_xy(waypoints.longitude, waypoints.latitude),
                                 crs='EPSG:4326')
    waypoints = waypoints.to_crs(2056)

    # Remove waypoints with low accuracy
    waypoints = waypoints[waypoints['accuracy'] <= ACCURACY]

    # Compute the distance between waypoints for each leg
    distances = pd.DataFrame()

    for i in waypoints.leg_id.unique():
        temp = waypoints[waypoints['leg_id'] == i].copy()
        temp['linestring'] = [(LineString([[a.x, a.y], [b.x, b.y]]) if b is not None else None)
                              for (a, b) in zip(temp.geometry, temp.geometry.shift(-1, axis=0))]
        if len(temp) > 1: temp.iloc[-1, -1] = temp.iloc[-2, -1]
        distances = pd.concat([distances, temp], axis=0)

    distances = gpd.GeoDataFrame(distances, geometry='linestring', crs='EPSG:2056')
    waypoints['distance'] = distances.length

    # Split trips if distance between 2 waypoints is too high
    waypoints['new_leg_id'] = ''

    for i in waypoints.leg_id.unique():
        temp = waypoints[waypoints['leg_id'] == i]
        count = 1
        if_loop = True
        for j, waypoint in temp.iterrows():
            if waypoint['distance'] >= DISTANCE:
                if not if_loop:
                    if_loop = True
                    waypoints.loc[j, 'new_leg_id'] = str(i) + '_' + str(count)
                    count += 1
            else:
                if_loop = False
                waypoints.loc[j, 'new_leg_id'] = str(i) + '_' + str(count)

    # Remove trips with only one waypoint
    waypoints = waypoints[waypoints.groupby('new_leg_id').new_leg_id.transform(len) > 1]
    waypoints = waypoints[waypoints['new_leg_id'] != '']

    # Create linestrings
    trips = waypoints.groupby(['new_leg_id'])['geometry'].apply(lambda x: LineString(x.tolist()))
    trips = gpd.GeoDataFrame(trips, geometry='geometry', crs='EPSG:2056')
    trips['leg_id'] = [trips.index[i][0:8] for i in range(len(trips))]
    trips['part'] = [trips.index[i][9] for i in range(len(trips))]

    # Aggregate linestrings by leg_id
    trips_agg = trips.groupby(['leg_id'])['geometry'].apply(lambda x: MultiLineString(x.tolist()))
    trips_agg = gpd.GeoDataFrame(trips_agg, geometry='geometry', crs='EPSG:2056')
    trips_agg['leg_id'] = trips_agg.index.values

    # Clip linestrings to cities buffer
    trips_clip = gpd.clip(trips_agg, buffer['geometry'])
    trips_clip = trips_clip[~trips_clip['geometry'].is_empty]

    # Assign cities to each linestring
    trips_clip['city'] = ''
    for i in range(len(buffer)):
        city = buffer.loc[i, 'AName']
        trips_clip.loc[trips_clip.intersects(buffer.loc[i, 'geometry']), 'city'] = city

    # Compute duration and length sum
    waypoints_clip = gpd.clip(waypoints, buffer['geometry'])
    waypoints_clip = waypoints_clip.sort_values(by=['leg_id', 'tracked_at'])
    waypoints_clip['duration_4sum'] = waypoints_clip.groupby('new_leg_id')['tracked_at'].diff().dt.total_seconds()
    trips_clip['length_comp'] = trips_clip.length
    trips_clip = trips_clip.set_index(trips_clip.index.values.astype(int))
    trips_clip['duration_comp'] = waypoints_clip.groupby('leg_id')['duration_4sum'].sum()

    # Add speed 
    trips_clip['speed'] = trips_clip['length_comp'] / trips_clip['duration_comp'] * 3600 / 1000  # km/h

    # Save DataFrame
    trips_clip.to_file("../pickles/linestrings/linestrings_" + c + ".geojson", driver='GeoJSON')


if __name__ == '__main__':
    main()
