"""
Title: Master’s project - Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

Content:
Regression models and plots

Dataset: MOBIS
Scale: Local, Multi-city

Author: Marine Manche
Date: 13/07/22
"""

import os
import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

sns.set(font_scale=1.4)
case = 'dur'

nature = ['forest', 'waterbank', 'park', 'other_green', 'vegetation_med', 'vegetation_high']
nature_title = ['Forest', 'Water bank', 'Park', 'Other green area', 'Partly treed street', 'Highly treed street']
cover = nature + ['grey', 'green', 'tot']
cover_title = nature_title + ['Grey area', 'Total green', 'Total green and grey']

cols = ['region', 'nature']
ses = ['edu_2', 'edu_3', 'age', 'gender']
city = ['city_Bern', 'city_Geneva', 'city_Lausanne', 'city_Zurich']


def linear_model(train_laus, train_swiss, test_laus, test_swiss, lm, case, socio=True):
    r2 = []
    r2_test = []

    m = MultiOutputRegressor(lm)

    # Lausanne
    X = train_laus[['exp_' + c + '_600' for c in nature] + (ses if socio else [])]
    y = train_laus[[case + '_' + c + '_20' for c in cover]]
    model = m.fit(X, y)

    for j, var in enumerate(cover): train_laus[var + '_model'] = model.predict(X)[:, j]
    for j, var in enumerate(cover): test_laus[var + '_model'] = model.predict(
        test_laus[['exp_' + c + '_600' for c in nature] + (ses if socio else [])])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_laus[case + '_' + var + '_20'], train_laus[var + '_model'])]
        r2_test = r2_test + [r2_score(test_laus[case + '_' + var + '_20'], test_laus[var + '_model'])]

    # Switzerland
    X = train_swiss[['exp_' + c for c in nature] + (ses if socio else []) + city]
    y = train_swiss[[case + '_' + c for c in cover]]
    model = m.fit(X, y)

    for j, var in enumerate(cover): train_swiss[var + '_model'] = model.predict(X)[:, j]
    for j, var in enumerate(cover): test_swiss[var + '_model'] = model.predict(
        test_swiss[['exp_' + c for c in nature] + (ses if socio else []) + city])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_swiss[case + '_' + var], train_swiss[var + '_model'])]
        r2_test = r2_test + [r2_score(test_swiss[case + '_' + var], test_swiss[var + '_model'])]

    return r2, r2_test


def cluster(train_laus, train_swiss, test_laus, test_swiss, clusters, case):
    r2 = []
    r2_test = []

    # Lausanne
    # K means clustering
    X = train_laus[['exp_' + c1 + '_' + c2 for c1 in nature for c2 in buff] + ses]
    m = KMeans(n_clusters=clusters, random_state=42)
    model = m.fit(X)
    train_laus['cluster'] = model.predict(X)

    # Calculating cluster
    X_test = test_laus[['exp_' + c1 + '_' + c2 for c1 in nature for c2 in buff] + ses]
    test_laus['cluster'] = model.predict(X_test)

    # Regression
    lm = MultiOutputRegressor(LinearRegression())

    X = train_laus[['cluster'] + ses]
    y = train_laus[[case + '_' + c + '_20' for c in cover]]
    model = lm.fit(X, y)
    for j, var in enumerate(cover): train_laus[var + '_model'] = model.predict(X)[:, j]
    for j, var in enumerate(cover): test_laus[var + '_model'] = model.predict(test_laus[['cluster'] + ses])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_laus[case + '_' + var + '_20'], train_laus[var + '_model'])]
        r2_test = r2_test + [r2_score(test_laus[case + '_' + var + '_20'], test_laus[var + '_model'])]

    # Switzerland
    # K means clustering
    X = train_swiss[['exp_' + c1 for c1 in nature] + ses + city]
    m = KMeans(n_clusters=clusters, random_state=42)
    model = m.fit(X)
    train_swiss['cluster'] = model.predict(X)

    # Calculating cluster
    X_test = test_swiss[['exp_' + c1 for c1 in nature] + ses + city]
    test_swiss['cluster'] = model.predict(X_test)

    # Regression
    lm = MultiOutputRegressor(LinearRegression())

    X = train_swiss[['cluster'] + ses]
    y = train_swiss[[case + '_' + c for c in cover]]
    model = lm.fit(X, y)
    for j, var in enumerate(cover): train_swiss[var + '_model'] = model.predict(X)[:, j]
    # if i in test['cluster'].values:
    for j, var in enumerate(cover): test_swiss[var + '_model'] = model.predict(test_swiss[['cluster'] + ses])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_swiss[case + '_' + var], train_swiss[var + '_model'])]
        r2_test = r2_test + [r2_score(test_swiss[case + '_' + var], test_swiss[var + '_model'])]

    return r2, r2_test


def regression_cluster(train_laus, train_swiss, test_laus, test_swiss, clusters, case, regressor, lr=False):
    r2 = []
    r2_test = []

    # Lausanne
    # K means clustering
    X = train_laus[['exp_' + c1 + '_' + c2 for c1 in nature for c2 in buff] + ses]
    m = KMeans(n_clusters=clusters, random_state=42)
    model = m.fit(X)
    train_laus['cluster'] = model.predict(X)

    # Calculating cluster
    X_test = test_laus[['exp_' + c1 + '_' + c2 for c1 in nature for c2 in buff] + ses]
    test_laus['cluster'] = model.predict(X_test)

    # Regression
    if lr:
        pipe = MultiOutputRegressor(regressor)
    else:
        pipe = Pipeline([('scale', MinMaxScaler()), ('reg', MultiOutputRegressor(regressor))])

    for i in range(clusters):
        train_clust = train_laus[train_laus['cluster'] == i]

        X = train_clust[['exp_' + c + '_600' for c in nature] + ses]
        y = train_clust[[case + '_' + c + '_20' for c in cover]]
        model = pipe.fit(X, y)
        for j, var in enumerate(cover): train_laus.loc[train_laus['cluster'] == i, var + '_model'] = model.predict(X)[:,
                                                                                                     j]
        for j, var in enumerate(cover): test_laus.loc[test_laus['cluster'] == i, var + '_model'] = model.predict(
            test_laus.loc[test_laus['cluster'] == i, ['exp_' + c + '_600' for c in nature] + ses])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_laus[case + '_' + var + '_20'], train_laus[var + '_model'])]
        r2_test = r2_test + [r2_score(test_laus[case + '_' + var + '_20'], test_laus[var + '_model'])]

    # Switzerland
    # K means clustering
    X = train_swiss[['exp_' + c1 for c1 in nature] + ses + city]
    m = KMeans(n_clusters=clusters, random_state=42)
    model = m.fit(X)
    train_swiss['cluster'] = model.predict(X)

    # Calculating cluster
    X_test = test_swiss[['exp_' + c1 for c1 in nature] + ses + city]
    test_swiss['cluster'] = model.predict(X_test)

    # Regression
    pipe = Pipeline([('scale', MinMaxScaler()), ('reg', MultiOutputRegressor(regressor))])

    for i in range(clusters):
        train_clust = train_swiss[train_swiss['cluster'] == i]

        X = train_clust[['exp_' + c for c in nature] + ses + city]
        y = train_clust[[case + '_' + c for c in cover]]
        model = pipe.fit(X, y)
        for j, var in enumerate(cover): train_swiss.loc[train_swiss['cluster'] == i, var + '_model'] = model.predict(X)[
                                                                                                       :, j]
        for j, var in enumerate(cover): test_swiss.loc[test_swiss['cluster'] == i, var + '_model'] = model.predict(
            test_swiss.loc[test_swiss['cluster'] == i, ['exp_' + c for c in nature] + ses + city])[:, j]

    for var in cover:
        r2 = r2 + [r2_score(train_swiss[case + '_' + var], train_swiss[var + '_model'])]
        r2_test = r2_test + [r2_score(test_swiss[case + '_' + var], test_swiss[var + '_model'])]

    return r2, r2_test


def main():
    # Load data
    os.system("git lfs pull -I 'pickles/encoded/users.pkl'")
    os.system("git lfs pull -I 'pickles/encoded/users_lausanne.pkl'")
    users_swiss = pkl.load(open("../pickles/encoded/users.pkl", "rb"))
    users_laus = pkl.load(open("../pickles/encoded/users_lausanne.pkl", "rb"))

    # Get SES dummies
    users_laus = pd.get_dummies(users_laus, columns=['edu'], drop_first=True)
    users_laus = users_laus.astype({'gender': bool, 'edu_2': bool, 'edu_3': bool})
    users_swiss = pd.get_dummies(users_swiss, columns=['edu'], drop_first=True)
    users_swiss = users_swiss.astype({'gender': bool, 'edu_2': bool, 'edu_3': bool})

    # Get cities dummies
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

    # Linear models
    summary_lm = pd.DataFrame(columns=cols)
    for region in ['Lausanne', 'Swiss']:
        for var in cover:
            summary_lm = pd.concat([summary_lm, pd.DataFrame([[region] + [var]], columns=cols)], ignore_index=True)

    summary_lm['linear'], summary_lm['linear_test'] = linear_model(train_laus=train_laus, train_swiss=train_swiss,
                                                                   test_laus=test_laus, test_swiss=test_swiss,
                                                                   lm=LinearRegression(), case=case, socio=False)
    summary_lm['linear_SES'], summary_lm['linear_SES_test'] = linear_model(train_laus=train_laus,
                                                                           train_swiss=train_swiss, test_laus=test_laus,
                                                                           test_swiss=test_swiss,
                                                                           lm=LinearRegression(), case=case)
    summary_lm['poly'], summary_lm['poly_test'] = linear_model(train_laus=train_laus, train_swiss=train_swiss,
                                                               test_laus=test_laus, test_swiss=test_swiss,
                                                               lm=Pipeline([('pf', PolynomialFeatures(degree=2)),
                                                                            ('lr', LinearRegression())]), case=case,
                                                               socio=False)
    summary_lm['poly_SES'], summary_lm['poly_SES_test'] = linear_model(train_laus=train_laus, train_swiss=train_swiss,
                                                                       test_laus=test_laus, test_swiss=test_swiss,
                                                                       lm=Pipeline(
                                                                           [('pf', PolynomialFeatures(degree=2)),
                                                                            ('lr', LinearRegression())]), case=case)
    summary_lm['pca'], summary_lm['pca_test'] = linear_model(train_laus=train_laus, train_swiss=train_swiss,
                                                             test_laus=test_laus, test_swiss=test_swiss,
                                                             lm=Pipeline([('pca', PCA(n_components=3)),
                                                                          ('lr', LinearRegression())]), case=case,
                                                             socio=False)
    summary_lm['gauss'], summary_lm['gauss_test'] = linear_model(train_laus=train_laus, train_swiss=train_swiss,
                                                                 test_laus=test_laus, test_swiss=test_swiss,
                                                                 lm=Pipeline([('gauss', PowerTransformer()),
                                                                              ('lr', LinearRegression())]), case=case,
                                                                 socio=False)

    # Machine learning models at the local scale
    summary_ml_laus = pd.DataFrame(columns=cols)
    for region in ['Lausanne', 'Swiss']:
        for var in cover:
            summary_ml_laus = pd.concat([summary_ml_laus, pd.DataFrame([[region] + [var]], columns=cols)],
                                        ignore_index=True)

    summary_ml_laus['clust'], summary_ml_laus['clust_test'] = cluster(train_laus=train_laus, train_swiss=train_swiss,
                                                                      test_laus=test_laus, test_swiss=test_swiss,
                                                                      clusters=2, case=case)
    summary_ml_laus['clust_linear'], summary_ml_laus['clust_linear_test'] = regression_cluster(train_laus=train_laus,
                                                                                               train_swiss=train_swiss,
                                                                                               test_laus=test_laus,
                                                                                               test_swiss=test_swiss,
                                                                                               clusters=2, case=case,
                                                                                               regressor=LinearRegression(),
                                                                                               lr=True)
    summary_ml_laus['rf'], summary_ml_laus['rf_test'] = regression_cluster(train_laus=train_laus,
                                                                           train_swiss=train_swiss, test_laus=test_laus,
                                                                           test_swiss=test_swiss, clusters=1, case=case,
                                                                           regressor=RandomForestRegressor(max_depth=10,
                                                                                                           max_features='sqrt',
                                                                                                           min_samples_leaf=4,
                                                                                                           min_samples_split=10,
                                                                                                           n_estimators=200,
                                                                                                           random_state=42))
    summary_ml_laus['xgboost'], summary_ml_laus['xgboost_test'] = regression_cluster(train_laus=train_laus,
                                                                                     train_swiss=train_swiss,
                                                                                     test_laus=test_laus,
                                                                                     test_swiss=test_swiss, clusters=1,
                                                                                     case=case,
                                                                                     regressor=XGBRegressor(
                                                                                         colsample_bytree=0.6, gamma=5,
                                                                                         max_depth=3,
                                                                                         min_child_weight=10,
                                                                                         subsample=0.8,
                                                                                         random_state=42))
    summary_ml_laus['clust_rf'], summary_ml_laus['clust_rf_test'] = regression_cluster(train_laus=train_laus,
                                                                                       train_swiss=train_swiss,
                                                                                       test_laus=test_laus,
                                                                                       test_swiss=test_swiss,
                                                                                       clusters=2, case=case,
                                                                                       regressor=RandomForestRegressor(
                                                                                           max_depth=10,
                                                                                           max_features='sqrt',
                                                                                           min_samples_leaf=4,
                                                                                           min_samples_split=10,
                                                                                           n_estimators=200,
                                                                                           random_state=42))

    # Machine learning models at the multi-city scale
    summary_ml_swiss = pd.DataFrame(columns=cols)
    for region in ['Lausanne', 'Swiss']:
        for var in cover:
            summary_ml_swiss = pd.concat([summary_ml_swiss, pd.DataFrame([[region] + [var]], columns=cols)],
                                         ignore_index=True)

    summary_ml_swiss['clust'], summary_ml_swiss['clust_test'] = cluster(train_laus=train_laus, train_swiss=train_swiss,
                                                                        test_laus=test_laus, test_swiss=test_swiss,
                                                                        clusters=2, case=case)
    summary_ml_swiss['clust_linear'], summary_ml_swiss['clust_linear_test'] = regression_cluster(train_laus=train_laus,
                                                                                                 train_swiss=train_swiss,
                                                                                                 test_laus=test_laus,
                                                                                                 test_swiss=test_swiss,
                                                                                                 clusters=2, case=case,
                                                                                                 regressor=LinearRegression())
    summary_ml_swiss['rf'], summary_ml_swiss['rf_test'] = regression_cluster(train_laus=train_laus,
                                                                             train_swiss=train_swiss,
                                                                             test_laus=test_laus, test_swiss=test_swiss,
                                                                             clusters=1, case=case,
                                                                             regressor=RandomForestRegressor(
                                                                                 max_depth=20, max_features='sqrt',
                                                                                 min_samples_leaf=4,
                                                                                 min_samples_split=10, n_estimators=300,
                                                                                 random_state=42))
    summary_ml_swiss['xgboost'], summary_ml_swiss['xgboost_test'] = regression_cluster(train_laus=train_laus,
                                                                                       train_swiss=train_swiss,
                                                                                       test_laus=test_laus,
                                                                                       test_swiss=test_swiss,
                                                                                       clusters=1, case=case,
                                                                                       regressor=XGBRegressor(
                                                                                           colsample_bytree=1, gamma=5,
                                                                                           max_depth=3,
                                                                                           min_child_weight=5,
                                                                                           subsample=1,
                                                                                           random_state=42))
    summary_ml_swiss['clust_rf'], summary_ml_swiss['clust_rf_test'] = regression_cluster(train_laus=train_laus,
                                                                                         train_swiss=train_swiss,
                                                                                         test_laus=test_laus,
                                                                                         test_swiss=test_swiss,
                                                                                         clusters=2, case=case,
                                                                                         regressor=RandomForestRegressor(
                                                                                             max_depth=20,
                                                                                             max_features='sqrt',
                                                                                             min_samples_leaf=4,
                                                                                             min_samples_split=10,
                                                                                             n_estimators=300,
                                                                                             random_state=42))

    # Merge and plot
    city = ['Lausanne', 'Swiss']
    city_title = [' at the local scale', ' at the multi-city scale']
    model_title = ['Regression models', 'Machine learning models']
    title = [model_title[0] + city_title[0], model_title[0] + city_title[1], model_title[1] + city_title[0],
             model_title[1] + city_title[1]]
    letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    fig, axn = plt.subplots(nrows=4, ncols=2, figsize=(17.5, 26), gridspec_kw={'height_ratios': [6, 6, 5, 5]},
                            sharex=True, sharey='row')

    cbar_ax = fig.add_axes([.93, .44, .02, .13])
    cmap = sns.color_palette("vlag", as_cmap=True)
    fig.supxlabel("Cover type", y=0.1)
    fig.supylabel("Model")

    y_labels_lm = ['LR', 'LR (SES)', 'PR', 'PR (SES)', 'PCA + LR', 'PT + LR']
    y_labels_ml = ['Clust', 'Clust + Acc', 'RF', 'XGBoost', 'Clust + RF']

    # Create merged DataFrame
    for i, ax in enumerate(axn.flat):
        if i < 4:
            df = summary_lm[summary_lm['region'] == city[int(i / 2)]]
        elif i < 6:
            df = summary_ml_laus[summary_ml_laus['region'] == city[0]]
        else:
            df = summary_ml_swiss[summary_ml_swiss['region'] == city[1]]

        df = df.set_index('nature')
        df = df.iloc[:, 2::2] if i % 2 else df.iloc[:, 1::2]
        df = df.astype(float).T

        # Plot
        if i < 4:
            sns.heatmap(df, ax=ax, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", square=True,
                        linewidths=0.5,
                        cbar=i == 0, cbar_ax=None if i else cbar_ax, cbar_kws={'ticks': [-1, -.5, 0, .5, 1]},
                        annot_kws={'size': 13})
            ax.set_yticklabels(y_labels_lm)


        else:
            sns.heatmap(df, ax=ax, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", square=True,
                        linewidths=0.5,
                        cbar=False, annot_kws={'size': 13})
            ax.set_yticklabels(y_labels_ml)

        ax.set_xlabel(None)
        ax.set_xticklabels(cover_title, rotation=45, ha='right')
        if i == 0: ax.set_title("Training set", y=1.1, fontweight="bold")
        if i == 1: ax.set_title("Testing set", y=1.1, fontweight="bold")
        if i % 2: ax.text(-0.08, 1.1, title[int(i / 2)], transform=ax.transAxes, va='top', ha='center')
        if not i % 2: ax.text(0.03, 1.11, letter[i], fontsize=24, transform=ax.transAxes, fontweight='bold', va='top')
        if i % 2: ax.text(0.92, 1.11, letter[i], fontsize=24, transform=ax.transAxes, fontweight='bold', va='top')

    plt.subplots_adjust(left=0.13, wspace=0.1, hspace=0.1, bottom=0.17)

    plt.savefig("../output/2_3_merged.pdf")
    plt.show()


if __name__ == '__main__':
    main()
