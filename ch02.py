import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, OneHotEncoder, CategoricalEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from encoder import CombinedAttributesAdder, DataFrameSelector, CategoriEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 50)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_idices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_idices[:test_set_size]
    train_indices = shuffled_idices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def display_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())


# fetch_housing_data()
housing = load_housing_data()
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# print(len(train_set), "train + ", len(test_set), "test")
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# print(housing["income_cat"].value_counts() / len(housing))

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idex, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idex]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"] / 100, label="population", figsize=(10, 7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))

housing_cat = housing["ocean_proximity"]
# cat_encoder = CategoricalEncoder()
# housing_cat_reshapee = housing_cat.values.reshape(-1, 1)
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshapee)

housing_cat_encoded, housing_categories = housing_cat.factorize()
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot.toarray())

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_addr', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs, numpy=True)),
    # ("ordinal_encoder", OrdinalEncoder()),
    ("cat_encoder", OneHotEncoder(sparse=False))
    # ('cat_encoder',  CategoriEncoder(cat_attribs))
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

# print(housing)
housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
#
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
print(np.sqrt(lin_mse))

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

print("================================================================")

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
print(np.sqrt(tree_mse))
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)

print("================================================================")

forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# print(np.sqrt(forest_mse))
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

print("================================================================")

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 4, 6, 8]}
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs + extra_attribs + housing_categories.values.tolist()
print(sorted(zip(feature_importances, attributes), reverse=True))

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_prediction = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_prediction)
print(np.sqrt(final_mse))
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
#
# column_labels = housing.columns.values
# column_labels = np.append(column_labels, [["a", "b"]])
# housing_temp = pd.DataFrame(housing_extra_attribs, columns=column_labels, index=list(housing.index.values))
# print(housing_temp.head())
# print(housing.index.values)

# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print(housing.head())
# housing.info()
# print(housing["ocean_proximity"].value_counts())

# print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()
