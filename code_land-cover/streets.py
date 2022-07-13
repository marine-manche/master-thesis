"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Extract OSM streets

Author: Marine Manche
Date: 13/07/22
"""

import osmnx as ox
import geopandas as gpd


def main():
    # Load cities' buffers
    polygon = gpd.read_file('input/swiss_cities_buffer_single_parts.geojson')
    polygon = polygon.to_crs(4326)
    geom = polygon['geometry'].values[0]

    streets = ['primary', 'secondary', 'tertiary', 'unclassified', 'residential']

    # Extract OSM layers
    landuse = ox.geometries_from_polygon(geom, tags={'highway': streets})
    landuse = landuse[['crossing', 'highway', 'geometry']]

    # Save OSM streets
    landuse.to_file("output/streets_osmnx_cols.geojson", driver='GeoJSON')


if __name__ == '__main__':
    main()
