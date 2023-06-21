import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_rsv (df1, input_test_size = 0.2, random_seed = 42):

    # 1. Select the features that are needed to be processed and how
    categorical_features = df1.select_dtypes(include=['category']).columns.tolist()
    categorical_features.remove('RSV_test_result')

    numeric_features_right = ['CCI', 'n_symptoms', 'prev_positive_rsv', 'previous_test_daydiff', 'n_immunodeficiencies']
    numeric_features_left = ['sine', 'cosine']

    # 2. Define transformers for every feature type and build the preprocessor
    # 2.1 First, dummy encode categoricals
    df1 = pd.get_dummies(df1, columns=categorical_features, drop_first= True)

    # 2.2. Second, transform the numerical features
    right_transformer = Pipeline(steps=[
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    left_transformer = Pipeline(steps=[
        ('exp', FunctionTransformer(np.exp, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_right', right_transformer, numeric_features_right),
            ('num_left', left_transformer, numeric_features_left)
        ])

    # 3. Transform the data
    X = df1.drop(['RSV_test_result'], axis=1)
    y = df1['RSV_test_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= input_test_size, random_state=random_seed)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test