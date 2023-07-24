# Data Load
from load import load_dataset_regression, load_dataset_binary_classification, load_dataset_multi_classification_woo, load_dataset_multi_classification_gang

# Regression
# from models.linear_regression import linear_regression
# from models.ridge import ridge
# from models.lasso import lasso
# from models.xgboost import xgboost
# from models.lightgbm import lightgbm
# from models.cnn import cnn_regression
from models.rnn_regression import rnn_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Binary Classification
from models.binary_model import binary_model

# Multi Classification
from models.random_forest_classifier import train_random_forest_classifier
from models.rnn_classification import rnn_classification
from sklearn.ensemble import RandomForestClassifier





# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report