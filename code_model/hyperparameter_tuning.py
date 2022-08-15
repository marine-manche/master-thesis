"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Hyperparameter tuning of random forest models

Dataset: MOBIS
Scale: Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import pickle as pkl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

XGBOOST = False
case = 'dur'


def main():

    # Load data
    os.system("git lfs pull -I 'pickles/encoded/users.pkl'")
    os.system("git lfs pull -I 'pickles/encoded/users_lausanne.pkl'")
    users_swiss = pkl.load(open("../pickles/encoded/users.pkl", "rb"))
    users_laus = pkl.load(open("../pickles/encoded/users_lausanne.pkl", "rb"))

    nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']
    cover = nature + ['grey', 'green', 'tot']

    # Get SES dummies
    ses = ['edu_2', 'edu_3', 'age', 'gender']
    users_laus = pd.get_dummies(users_laus, columns=['edu'], drop_first=True)
    users_laus = users_laus.astype({'gender': bool, 'edu_2': bool, 'edu_3': bool})
    users_swiss = pd.get_dummies(users_swiss, columns=['edu'], drop_first=True)
    users_swiss = users_swiss.astype({'gender': bool, 'edu_2': bool, 'edu_3': bool})

    # Get cities dummies
    city = ['city_Bern', 'city_Geneva', 'city_Lausanne', 'city_Zurich']
    users_swiss = pd.get_dummies(users_swiss, columns=['city'], drop_first=True)
    users_swiss = users_swiss.astype(
        {'city_Bern': bool, 'city_Geneva': bool, 'city_Lausanne': bool, 'city_Zurich': bool})

    # Create training and testing sets
    rand_df = users_laus.sample(len(users_laus), random_state=26)
    train_laus = rand_df[:int(0.8 * len(rand_df))].copy()
    test_laus = rand_df[int(0.8 * len(rand_df)):].copy()

    rand_df = users_swiss.sample(len(users_swiss), random_state=26)
    train_swiss = rand_df[:int(0.8 * len(rand_df))].copy()
    test_swiss = rand_df[int(0.8 * len(rand_df)):].copy()

    # Model

    r2 = []
    r2_test = []

    # With XGBoost algorithm
    if XGBOOST:
        # Define Grid
        grid = {
            'estimator__min_child_weight': [1, 5, 10],
            'estimator__gamma': [0.5, 1, 1.5, 2, 5],
            'estimator__subsample': [0.6, 0.8, 1.0],
            'estimator__colsample_bytree': [0.6, 0.8, 1.0],
            'estimator__max_depth': [3, 4, 5],
            'estimator__random_state': [42]
        }

        # Model
        m = MultiOutputRegressor(XGBRegressor())
        clf = GridSearchCV(estimator=m, param_grid=grid, cv=5)

    # With Random Forest Regressor
    else:
        # Define Grid
        grid = {
            'estimator__bootstrap': [True, False],
            'estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'estimator__max_features': ['sqrt', 'log2'],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__n_estimators': [100, 200, 300, 400],
            'estimator__random_state': [42]
        }

        # Model
        m = MultiOutputRegressor(RandomForestRegressor())
        clf = GridSearchCV(estimator=m, param_grid=grid, cv=5)

    # Local scale
    X = train_laus[['exp_' + c + '_600' for c in nature] + ses]
    y = train_laus[[case + '_' + c + '_20' for c in cover]]
    model = clf.fit(X, y)
    print(clf.best_params_)
    for j, var in enumerate(cover): train_laus[var + '_model'] = model.predict(X)[:, j]
    for j, var in enumerate(cover): test_laus[var + '_model'] = model.predict(
        test_laus[['exp_' + c + '_600' for c in nature] + ses])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_laus[case + '_' + var + '_20'], train_laus[var + '_model'])]
        r2_test = r2_test + [r2_score(test_laus[case + '_' + var + '_20'], test_laus[var + '_model'])]

    # Multi-city scale
    X = train_swiss[['exp_' + c for c in nature] + ses + city]
    y = train_swiss[[case + '_' + c for c in cover]]
    model = clf.fit(X, y)
    print(clf.best_params_)
    for j, var in enumerate(cover): train_swiss[var + '_model'] = model.predict(X)[:, j]
    for j, var in enumerate(cover): test_swiss[var + '_model'] = model.predict(
        test_swiss[['exp_' + c for c in nature] + ses + city])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_swiss[case + '_' + var], train_swiss[var + '_model'])]
        r2_test = r2_test + [r2_score(test_swiss[case + '_' + var], test_swiss[var + '_model'])]

    df = pd.DataFrame(data={'R2 train': r2, 'R2 test': r2_test})

    # Save parameters
    if XGBOOST:
        pkl.dump(df, open("../pickles/model/xgboost_search_grid.pkl", "wb"))
    else:
        pkl.dump(df, open("../pickles/model/rand_forest_search_grid.pkl", "wb"))


if __name__ == '__main__':
    main()
