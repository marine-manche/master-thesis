"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Format users variables

Dataset: CoLaus
Scale: Local

Author: Marine Manche
Date: 13/07/22
"""

import os
import pickle as pkl


def main():
    # Load files
    os.system("git lfs pull -I 'pickles/colaus/colaus_exp.pkl'")
    colaus = pkl.load(open("../pickles/colaus/colaus_exp.pkl", "rb"))

    colaus.insert(6, 'mvpa', colaus['F2dur_day_mod'] + colaus['F2dur_day_vig'])  # Compute mvpa
    colaus = colaus[colaus['mvpa'].notna()]

    colaus = colaus.rename(columns={'pt': 'ID', 'edtyp3': 'edu', 'F2sex': 'gender', 'F2age': 'age', 'F2BMI': 'BMI',
                                    'F2dur_day_mod': 'mod_pa', 'F2dur_day_vig': 'vig_pa', 'F2gaf_l': 'gaf_l',
                                    'F2gaf_w': 'gaf_w', 'F2gaf_c': 'gaf_c', 'F2STs_tot': 'STs_tot'})  # Rename columns
    colaus = colaus.drop(columns={'F2dur_mvpa_d10'})  # Remove NaN

    # Add geometry columns
    colaus.insert(11, 'x', colaus["geometry"].apply(lambda p: p.x))
    colaus.insert(12, 'y', colaus["geometry"].apply(lambda p: p.y))

    # Save file
    pkl.dump(colaus, open("../pickles/encoded/colaus.pkl", "wb"))


if __name__ == '__main__':
    main()
