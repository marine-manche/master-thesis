"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Extract OSM land cover layers

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

    lu = ['allotments', 'cemetery', 'forest', 'grass', 'meadow', 'orchard', 'recreation_ground', 'vineyard', 'grass']
    leisures = ['garden', 'park', 'pitch']
    nature = ['wood', 'water', 'beach']

    # Extract OSM layers
    landuse = ox.geometries_from_polygon(geom, tags={'landuse': lu, 'leisure': leisures, 'natural': nature})
    landuse = landuse[['name', 'leisure', 'sport', 'natural', 'landuse', 'water', 'geometry']]

    # Save OSM layers
    landuse.to_file("../output/landuse_osmnx_cols.geojson", driver='GeoJSON')


if __name__ == '__main__':
    main()
