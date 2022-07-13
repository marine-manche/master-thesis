"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Smooth linestrings

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import movingpandas as mpd
import pickle as pkl
import string


def main():
    for c in string.ascii_uppercase:

        # Load waypoints
        waypoints = pkl.load(open("../pickles/waypoints_cities/waypoints_" + c + ".pkl", "rb"))

        waypoints = waypoints.rename(columns={'speed': 'speed_mobis'})

        # Create trajectory collection
        traj_collection = mpd.TrajectoryCollection(waypoints, 'leg_id', t='tracked_at', x='longitude', y='latitude')
        traj_collection.add_speed()

        # Smooth trajectories
        smooth = mpd.trajectory_smoother.KalmanSmootherCV(traj_collection).smooth(process_noise_std=0.5,
                                                                                  measurement_noise_std=10)
        smooth = smooth.to_point_gdf()

        # Save DataFrame
        pkl.dump(smooth, open("../pickles/waypoints_smoothed/waypoints_" + c + ".pkl", "wb"))


if __name__ == '__main__':
    main()
