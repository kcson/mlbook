from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

rooms_ix, bedrooms_ix, population_ix, houehold_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, houehold_ix]
        population_per_household = X[:, population_ix] / X[:, houehold_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, numpy=True):
        self.attribute_names = attribute_names
        self.numpy = numpy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.numpy:
            return X[self.attribute_names].values
        else:
            return X[self.attribute_names]


class CategoriEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        housing_cat_encoded, housing_categories = X["ocean_proximity"].factorize()
        print(housing_categories)
        encoder = OneHotEncoder()
        housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

        return housing_cat_1hot
