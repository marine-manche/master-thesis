"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Keep linestrings longer than 5 minutes and faster than 2 km/h
Compute share of nature for each nature type in users' trips

Dataset: MOBIS
Scale: Local

Author: Marine Manche
Date: 13/07/22
"""

import geopandas as gpd
import string

TIME = 5 * 60  # [s]
SPEED = 2


def main():
    for c in string.ascii_uppercase:
        # Load land cover data
        lulc = gpd.read_file("../data/geodata/land_use_geom_fixed.geojson")
        cities = gpd.read_file('../data/geodata/swiss_cities_buffer_single_parts.geojson')  # Swiss cities buffer

        # Add linestrings geometry to legs
        legs = gpd.read_file('../pickles/linestrings/linestrings_' + c + '.geojson')

        # Select legs in Lausanne
        legs = legs[legs['city'] == 'Lausanne']

        # Keep linestrings longer than 5 minutes and faster than 2 km/h
        legs = legs[(legs['duration_comp'] > TIME) & (legs['speed'] > SPEED)]

        nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']
        cover = nature + ['green', 'grey']

        for BUFFER in [10, 20]:
            for var in cover:
                legs['use_' + var + '_' + str(BUFFER)] = 0

            # Apply a buffer around linestrings and clip to nature types
            for i, leg in legs.iterrows():
                # Create buffer and clip it to nature types
                buff = leg.geometry.buffer(BUFFER)
                buff = gpd.clip(gpd.GeoDataFrame(index=[0], crs='epsg:2056', geometry=[buff]),
                                cities['geometry'])  # Clip to Lausanne buffer
                area = buff.area[0]
                leg_cliped = gpd.clip(lulc, buff)

                # Share of nature type
                add = 0
                if not leg_cliped.empty:
                    leg_cliped['area'] = leg_cliped.area
                    tab = leg_cliped.groupby(['category'])['area'].sum()

                    for var in nature:
                        temp = 0
                        if var in tab.index:
                            temp = tab.loc[var]
                            legs.loc[i, 'use_' + var + '_' + str(BUFFER)] = temp / area
                        add += temp

                # Get the total green share
                legs.loc[i, 'use_green_' + str(BUFFER)] = add / area

                # Get the grey share
                legs.loc[i, 'use_grey_' + str(BUFFER)] = (area - add) / area

        # Save DataFrame
        legs.to_file("../pickles/linestrings_use_lausanne/linestrings_" + c + ".geojson", driver='GeoJSON')


if __name__ == '__main__':
    main()
