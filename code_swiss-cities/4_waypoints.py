"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Extract legs of users of interest
Select legs done by walk or bicycle

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import string
import pandas as pd
import pickle as pkl


def main():
    users = pkl.load(open("../pickles/users_cities.pkl", "rb"))

    # Provide geometry of start and end point of each leg
    os.system("git lfs pull -I 'data/aggregated-mobis-2/mobis_legs.csv'")
    legs = pd.read_csv("../data/aggregated-mobis-2/mobis_legs.csv")

    # Select legs by walk or bicycle of users in cities 
    legs = legs[legs['user_id'].isin(users['user_id'])]
    legs = legs[(legs['mode'] == 'Mode::Walk') | (legs['mode'] == 'Mode::Bicycle')]

    # Save DataFrame
    pkl.dump(legs, open("../pickles/legs_cities.pkl", "wb"))

    # Change date format
    legs['started_at'] = pd.to_datetime(legs['started_at'])
    legs['finished_at'] = pd.to_datetime(legs['finished_at'])

    # Load waypoints and assign legs
    for c in string.ascii_uppercase[13:]:
        print(c)
        os.system("git lfs pull -I 'pickles/waypoints/waypoints_" + c + ".pkl'")
        waypoints = pkl.load(open("../pickles/waypoints/waypoints_" + c + ".pkl", "rb"))

        waypoints['tracked_at'] = pd.to_datetime(waypoints['tracked_at'])
        waypoints['leg_id'] = ''
        print(len(waypoints))

        legs_temp = legs[[char == c for char in [legs['user_id'].values[i][0] for i in range(len(legs))]]]

        for i, leg in legs_temp.iterrows():
            user = leg['user_id']
            leg_id = leg['trip_id']
            start = leg['started_at']
            end = leg['finished_at']

            # Assign leg_id to corresponding waypoints
            waypoints.loc[(waypoints['tracked_at'] >= start) & (waypoints['tracked_at'] <= end)
                          & (waypoints['user_id'] == user), 'leg_id'] = leg_id

        # Removing waypoints not to linked to legs of interest
        waypoints = waypoints[(waypoints['leg_id'] != '')]
        print(len(waypoints))

        # Removing waypoints with a leg of only 1 point
        waypoints = waypoints[waypoints.groupby('leg_id').leg_id.transform(len) > 1]
        print(len(waypoints))

        # Save waypoints with leg_id
        pkl.dump(waypoints, open("../pickles/waypoints_cities/waypoints_" + c + ".pkl", "wb"))


if __name__ == '__main__':
    main()
