"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Add linestrings geometry to legs

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import string
import pandas as pd
import pickle as pkl
import geopandas as gpd


def main():
    # Load data
    os.system("git lfs pull -I 'pickles/legs_cities.pkl'")
    legs_start = pkl.load(open("../pickles/legs_cities.pkl", "rb"))

    legs_linestrings = pd.DataFrame()

    for c in string.ascii_uppercase:
        # Load data
        os.system("git lfs pull -I 'pickles/linestrings_use_vegetation/linestrings_" + c + ".geojson'")
        linestrings = gpd.read_file("../pickles/linestrings_use_vegetation/linestrings_" + c + ".geojson")

        linestrings = linestrings.rename(columns={'leg_id': 'trip_id'})
        linestrings['trip_id'] = linestrings['trip_id'].astype(int)

        # Add linestrings geometry to legs
        legs_geom = pd.merge(legs_start, linestrings, how="left", on='trip_id')
        legs_gdf = gpd.GeoDataFrame(legs_geom, geometry=legs_geom.geometry, crs='EPSG:2056')

        # Remove legs with no geometry
        legs = legs_gdf[legs_gdf['geometry'] is not None]

        legs_linestrings = pd.concat([legs_linestrings, legs], axis=0)

    # Save Data Frame
    pkl.dump(legs_linestrings, open("../pickles/legs_use_vegetation.pkl", "wb"))


if __name__ == '__main__':
    main()
