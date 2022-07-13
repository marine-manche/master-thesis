"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Format users and legs variables
Assign city to each user
Add length of tracking study
Select legs in users' cities
Compute mean length and duration, median speed for each user
Compute mean of duration, distance and share of choice to use for each nature type at user scale

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import pandas as pd
import pickle as pkl
import geopandas as gpd


def main():
    nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']
    cover = nature + ['green', 'grey']

    # Users data
    os.system("git lfs pull -I 'pickles/users_exp.pkl'")
    users = pkl.load(open("../pickles/users_exp.pkl", "rb"))
    buffer = gpd.read_file('../data/geodata/swiss_cities_buffer_single_parts.geojson')

    # Assign cities to each user
    users['city'] = ''
    for i in range(len(buffer)):
        city = buffer.loc[i, 'AName']
        users.loc[users.intersects(buffer.loc[i, 'geometry']), 'city'] = city

    # Modify education & gender column
    users = users.replace({'education': {'Higher education (e.g., university)': 1,
                                         'Secondary education (e.g., apprenticeship or diploma)': 2,
                                         'Mandatory education': 3},
                           'gender': {'Male': 1, 'Female': 0}})
    users = users.rename(columns={'education': 'edu'})

    # Modify citizen column
    users['citsws'] = 0
    users.loc[users['citizen_1'] == 'Switzerland', 'citsws'] = 1

    baseline = ['user_id', 'citsws', 'edu', 'gender', 'age', 'language', 'main_employment', 'income', 'household_size',
                'city', 'geometry']
    users = users[baseline + ['exp_' + c for c in nature + ['green']]]

    # Add length of tracking study
    os.system("git lfs pull -I 'data/user_treatment_days_recorded.csv'")
    days = pd.read_csv("../data/user_treatment_days_recorded.csv")

    days['day_count'] = days[['day_count_phase_1', 'day_count_phase_2', 'day_count_phase_3']].sum(axis=1)

    users['day_count'] = 0
    for i, user in users.iterrows():
        count = days.loc[days['user_id'] == user.user_id, 'day_count'].values
        users.loc[i, 'day_count'] = count

    # Legs data
    os.system("git lfs pull -I 'pickles/legs_use_vegetation.pkl'")
    legs = pkl.load(open("../pickles/legs_use_vegetation.pkl", "rb"))

    # Keep legs only in user's city
    legs['user_city'] = ''
    for i, leg in legs.iterrows():
        city = users.loc[users['user_id'] == leg.user_id, 'city'].values[0]
        legs.loc[i, 'user_city'] = city
    legs = legs[legs['city'] == legs['user_city']]

    # Keep users with legs recorded
    users = users[users['user_id'].isin(legs.user_id.unique())]

    # Keep legs and users of only bigger cities
    users = users[users['city'].isin(['Zurich', 'Lausanne', 'Bern', 'Geneva', 'Basel'])]
    legs = legs[legs['user_city'].isin(['Zurich', 'Lausanne', 'Bern', 'Geneva', 'Basel'])]

    # Encode time of the day
    legs['started_at'] = pd.to_datetime(legs['started_at'])
    legs['finished_at'] = pd.to_datetime(legs['finished_at'])

    legs.loc[legs['mode'] == 'Mode::Bicycle', 'mode'] = 'bicycle'
    legs.loc[legs['mode'] == 'Mode::Walk', 'mode'] = 'walk'
    legs = legs.drop(columns=['detected_mode', 'was_confirmed', 'in_switzerland', 'labeled_purpose', 'length',
                              'duration'])
    legs = legs.rename(columns={'length_comp': 'length', 'imputed_purpose': 'purpose', 'duration_comp': 'duration'})
    legs = legs[['user_id', 'trip_id', 'treatment', 'phase', 'started_at', 'finished_at', 'length', 'duration', 'speed',
                 'mode', 'purpose'] + ['use_' + c for c in cover] + ['geometry']]

    # Compute average length, time and speed of activity 
    users['mean_length'] = 0
    users['mean_duration'] = 0
    users['med_speed'] = 0
    users['length'] = 0
    users['duration'] = 0

    for var in cover:
        users['dist_' + var] = 0
        users['dur_' + var] = 0
        users['use_' + var] = 0

    for user in users['user_id']:
        # Compute means of activity variables
        users.loc[(users['user_id'] == user), 'mean_length'] = legs.loc[legs['user_id'] == user, 'length'].mean()
        users.loc[(users['user_id'] == user), 'mean_duration'] = legs.loc[legs['user_id'] == user, 'duration'].mean()
        users.loc[(users['user_id'] == user), 'med_speed'] = legs.loc[legs['user_id'] == user, 'speed'].median()
        users.loc[(users['user_id'] == user), 'length'] \
            = legs.loc[legs['user_id'] == user, 'length'].sum() / users.loc[users['user_id'] == user,
                                                                            'day_count']  # [m/d]
        users.loc[(users['user_id'] == user), 'duration'] \
            = legs.loc[legs['user_id'] == user, 'duration'].sum() / 60 / users['day_count']  # [min/d]

        # Compute means of nature types
        for var in cover:
            users.loc[(users['user_id'] == user), 'dist_' + var] \
                = ((legs.loc[legs['user_id'] == user, 'use_' + var]
                    * legs.loc[legs['user_id'] == user, 'length']).sum()) / users.loc[users['user_id'] == user,
                                                                                      'day_count']  # [m/d]
            users.loc[(users['user_id'] == user), 'dur_' + var] \
                = ((legs.loc[legs['user_id'] == user, 'use_' + var]
                    * legs.loc[legs['user_id'] == user, 'duration']).sum()) / 60 / users['day_count']  # [min/d]
            users.loc[(users['user_id'] == user), 'use_' + var] \
                = users.loc[(users['user_id'] == user), 'dist_' + var] / users.loc[(users['user_id'] == user), 'length']

    # Encode totals
    for i in ['dist', 'dur', 'use']: users[i + '_tot'] = users[i + '_green'] + users[i + '_grey']

    # Format values to numeric
    cols = users.columns.drop(baseline)
    users[cols] = users[cols].apply(pd.to_numeric, errors='coerce')

    # Save DataFrames
    pkl.dump(legs, open("../pickles/encoded/legs.pkl", "wb"))
    pkl.dump(users, open("../pickles/encoded/users.pkl", "wb"))


if __name__ == '__main__':
    main()
