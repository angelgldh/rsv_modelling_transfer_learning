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
from category_encoders import OneHotEncoder


def preprocess_rsv (df1, input_test_size = 0.2, random_seed = 42):

    # 1. Select the features that are needed to be processed and how
    categorical_features = df1.select_dtypes(include=['category']).columns.tolist()
    categorical_features.remove('RSV_test_result')

    numeric_features_right = ['CCI', 'n_symptoms', 'prev_positive_rsv', 'previous_test_daydiff', 'n_immunodeficiencies']
    numeric_features_left = ['sine', 'cosine']

    # 2. Define transformers for every feature type and build the preprocessor
    categorical_transformer = OneHotEncoder(drop_invariant=True, )

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
            ('cat', categorical_transformer, categorical_features),
            ('num_right', right_transformer, numeric_features_right),
            ('num_left', left_transformer, numeric_features_left)
        ])

    # 3. Transform the data
    X = df1.drop(['RSV_test_result'], axis=1)
    y = df1['RSV_test_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= input_test_size, random_state=random_seed)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test, preprocessor

def train_model_rsv(model, param_grid, target_scorer, n_cv_folds,
                    X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=target_scorer, cv=n_cv_folds)

    print(f'Training model ... {model}')

    grid_search.fit(X_train, y_train)

    print('Best training parameters: ', grid_search.best_params_)
    print('Best training f1-score: ', grid_search.best_score_)

    return grid_search



def calculate_performance_metrics_rsv(trained_model, X_test, y_test, print_roc = True):
    # 1. First, compute predictions of the model, both pointwise and in probabilities
    y_pred = trained_model.predict(X_test)
    y_probs = trained_model.predict_proba(X_test)[:, 1]

    # 2. Calculate the confusion matrix metrics
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = recall
    specificity = tn / (tn + fp)
    ppv = precision
    npv = tn / (tn + fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    f1 = (2*precision*recall)/(precision + recall)

    # 3. Calculate the roc curve
    aux_y_test = [1 if label == 'Positive' else 0 for label in y_test]
    fpr, tpr, thresholds = roc_curve(aux_y_test, y_probs)

    auc_score = roc_auc_score(aux_y_test, y_probs)

    # 4. Print metrics and (if requested) the ROC Curve
    print(f'AUC Score: {auc_score}')
    print(f'Precision / Positive predictive value: {precision}')
    print(f'Specificity: {specificity}')
    print(f'Recall / sensitivity: {recall}')
    print(f'Negative predictive value: {npv}')
    print(f'Accuracy: {accuracy}')
    print(f'F-1: {f1}')

    if print_roc:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return auc_score, precision, recall, specificity, npv, accuracy, f1