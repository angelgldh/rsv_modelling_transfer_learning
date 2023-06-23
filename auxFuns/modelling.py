import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def preprocess_rsv (df1, input_test_size = 0.2, random_seed = 42):
    """
    Preprocesses the input data for the RSV phase 1 modelling stage, by applying feature 
    transformations and splitting the data into training and testing sets.

    Parameters:
    - df1 (DataFrame): The input DataFrame containing the data to be processed.
    - input_test_size (float): The proportion of the data to be used for testing. Defaults to 0.2.
    - random_seed (int): The random seed value for reproducibility. Defaults to 42.

    Returns:
    - X_train_transformed (ndarray): Transformed training features.
    - y_train (pd.Series): Training labels.
    - X_test_transformed (ndarray): Transformed testing features.
    - y_test (pd.Series): Testing labels.
    - preprocessor (ColumnTransformer): The preprocessor object used for feature transformations.
    """

    # 1. Select the features that are needed to be processed and how
    categorical_features = df1.select_dtypes(include=['category']).columns.tolist()
    categorical_features.remove('RSV_test_result')
    categorical_features.remove('calendar_year') # the reason behind this is that we will introduce manually the categories for calendar_year

    numeric_features_right = ['CCI', 'n_symptoms', 'prev_positive_rsv', 'previous_test_daydiff', 'n_immunodeficiencies']
    numeric_features_left = ['sine', 'cosine']

    # 2. Define transformers for every feature type and build the preprocessor
    # 2.1 Categorical features first
    categorical_transformer = OneHotEncoder(drop = 'first')
    calendar_year_transformer = OneHotEncoder(categories= [sorted(list(df1['calendar_year'].unique()))] , drop = 'first')

    # 2.2. Numeric features second
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
            ('cal_year',calendar_year_transformer, ['calendar_year']),
            ('num_right', right_transformer, numeric_features_right),
            ('num_left', left_transformer, numeric_features_left)
        ])

    # 3. Transform the data
    X = df1.drop(['RSV_test_result'], axis=1)
    y = df1['RSV_test_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= input_test_size, random_state=random_seed, stratify=y)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test, preprocessor


def preprocess_and_resample_rsv (df1, input_test_size = 0.2, random_seed = 42, 
                             resampling_technique = None, ratio_maj_min = 0.8):
    """
    Preprocesses the input data for the RSV phase 1 modelling stage, by applying feature 
    transformations and splitting the data into training and testing sets.

    Parameters:
    - df1 (DataFrame): The input DataFrame containing the data to be processed.
    - input_test_size (float): The proportion of the data to be used for testing. Defaults to 0.2.
    - random_seed (int): The random seed value for reproducibility. Defaults to 42.

    Returns:
    - X_train_transformed (ndarray): Transformed training features.
    - y_train (pd.Series): Training labels.
    - X_test_transformed (ndarray): Transformed testing features.
    - y_test (pd.Series): Testing labels.
    - preprocessor (ColumnTransformer): The preprocessor object used for feature transformations.
    """

    # 0. To avoid data leakage, first thing to do is split data into train and test
    X = df1.drop(['RSV_test_result'], axis=1)
    y = df1['RSV_test_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= input_test_size, random_state=random_seed,
                                                        stratify= y)
    
    # 1. Select the features that are needed to be processed and how
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    # categorical_features.remove('RSV_test_result')
    categorical_mask = X_train.columns.isin(categorical_features)
    categorical_features.remove('calendar_year') # the reason behind this is that we will introduce manually the categories for calendar_year
    categorical_features.remove('sex') # the reason behind this is that we will introduce manually the categories for sex

    numeric_features_right = ['CCI', 'n_symptoms', 'prev_positive_rsv', 'previous_test_daydiff', 'n_immunodeficiencies']
    numeric_features_left = ['sine', 'cosine']

    # 2. Add resampling of the data
    print("Resampling method chosen:")

    if resampling_technique == "over":
        print("\nOversampling")
        resampler = RandomOverSampler(sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique == "under":
        print("\nUndersampling")
        resampler = RandomUnderSampler(sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique == "smotenc":
        print("\nSMOTE-sampling")
        resampler = SMOTENC(categorical_features = categorical_mask,
                            sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique is None:
        print("\nNone")

    else:
        raise ValueError("Please, indicate a resampling technique. Accepted values are ['over', 'under', 'smotenc', None]")
    
    # 2.1 Fit the resampler to the data
    if resampling_technique is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)


    # 3. Define transformers for every feature type and build the preprocessor
    # 3.1 Categorical features first
    categorical_transformer = OneHotEncoder(drop = 'first')
    calendar_year_transformer = OneHotEncoder(categories= [sorted(list(df1['calendar_year'].unique()))] , drop = 'first')
    sex_transformer = OneHotEncoder(categories= [['Unknown', 'F', 'M']] , drop = 'first')

    # 3.2. Numeric features second
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
            ('cal_year',calendar_year_transformer, ['calendar_year']),
            ('se', sex_transformer, ['sex']),
            ('num_right', right_transformer, numeric_features_right),
            ('num_left', left_transformer, numeric_features_left)
        ])

    # 4. Finally, transform the data

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test, preprocessor



def train_model_rsv(model, param_grid, target_scorer, n_cv_folds,
                    X_train, y_train):
    """
    Trains a model for the RSV phase 1 modelling stage using grid search and cross-validation.

    Parameters:
    - model (object): The model object or estimator to be trained.
    - param_grid (dict): The dictionary of parameter grid for grid search.
    - target_scorer (str oor callable): The scoring metric for evaluating the model.
    - n_cv_folds (ind): The number of cross-validation folds.
    - X_train (nd-array): The training features (array-like or sparse matrix).
    - y_train (pd.series): The training labels (array-like).

    Returns:
    - grid_search (GridSearchCV): The trained grid search object.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=target_scorer, cv=n_cv_folds)

    print(f'Training model ... {model}')

    grid_search.fit(X_train, y_train)

    print('Best training parameters: ', grid_search.best_params_)
    print('Best training f1-score: ', grid_search.best_score_)

    return grid_search



def find_optimal_moving_threshold(model, X_test, y_test):
    """
    This function finds the optimal threshold for a binary classifier that maximizes the F1 score

    Parameters:
    - model (object): The trained model object.
    - X_test (nd-array): The testing features.
    - y_test (pd.Series): The testing labels.

    Returns:
    - optimal_threshold (float): The optimal threshold that maximizes F1 score.
    """
    aux_y_test = [1 if label == 'Positive' else 0 for label in y_test]


    y_probs = model.predict_proba(X_test)[:, 1]  # get the predicted probabilities for positive class

    thresholds = np.arange(0, 1, 0.01)  # generate a range of possible thresholds

    f1_scores = [f1_score(y_true= aux_y_test, y_pred = ([1 if y > threshold else 0 for y in y_probs])) for threshold in thresholds]

    optimal_threshold = thresholds[np.argmax(f1_scores)]  # find threshold that maximized F1 score
    optimal_f1 = np.max(f1_scores)

    print(f'Optimal threshold: {optimal_threshold}')
    print(f'Optimal f1: {optimal_f1}')
    print('\n')

    return optimal_threshold

def calculate_performance_metrics_rsv(trained_model, X_test, y_test, threshold= 0.5, print_roc = True):
    """
    Calculates performance metrics for the RSV phase 1 modelling stage based on the trained model's predictions.

    Parameters:
    - trained_model (object): The trained model object.
    - X_test (nd-array): The testing features.
    - y_test (pd.Series): The testing labels.
    - threshold (int): decision threshold for the binary classification
    - print_roc (boolean): Whether to print the ROC curve. Defaults to True.

    Returns:
    - auc_score (float): The Area Under the ROC Curve .
    - precision (float): Precision or Positive Predictive Value.
    - recall (float): Recall or Sensitivity.
    - specificity (float): Specificity.
    - npv (float): Negative Predictive Value.
    - accuracy (float): Accuracy.
    - f1 (float): F1-score.
    """

    # 1. First, compute predictions of the model, both pointwise and in probabilities
    # y_pred = trained_model.predict(X_test)
    y_probs = trained_model.predict_proba(X_test)[:, 1]
    y_pred = ["Positive" if p > threshold else "Negative"  for p in y_probs]

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
    f1_aux = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true= y_test, y_pred=y_pred, pos_label= "Positive")

    if f1_aux != f1:
        raise ValueError("F1_aux and F1 scores do not coincide.")

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


def get_feature_names_OneHotEncoder_preprocessor(column_transformer):
    """
    Retrieves the feature names from the OneHotEncoder-based preprocessor in a ColumnTransformer.

    Parameters:
    - column_transformer (ColumnTransformer): The ColumnTransformer object containing the OneHotEncoder-based preprocessor.

    Returns:
    - feature_names (list): feature names generated by the OneHotEncoder-based preprocessor.
    """
    feature_names = []

    # Loop over all transformers to get their names
    for name, trans, columns in column_transformer.transformers_:
        if isinstance(trans, Pipeline): 
            trans = trans.steps[-1][1]
        if isinstance(trans, OneHotEncoder):
            # if transformer is one hot encoder, we get names from get_feature_names_out
            feature_names.extend(list(trans.get_feature_names_out(columns)))
        elif isinstance(trans, StandardScaler):
            # if transformer is standard scalar, we just take column names
            feature_names.extend(columns)

    return feature_names


def plot_feature_importance_rf_rsv(preprocessor_rsv, trained_model):
    """
    Plots the feature importances for the Random Forest model used in the RSV modelling phase 1.

    Parameters:
    - preprocessor_rsv (ColumnTransformer): The preprocessor object used for feature transformations.
    - trained_model (GridSearchCV): The trained Random Forest model object.

    Returns:
    - None
    """

    # As the preprocessing of data transformed the features names too, it is needed to retrieve the new names 
    transformed_feature_names = get_feature_names_OneHotEncoder_preprocessor(preprocessor_rsv)

    # Get feature importances
    importances = trained_model.best_estimator_.feature_importances_

    # Create a feature importance dataframe and sort in ascending order
    importance_df = pd.DataFrame({'feature': transformed_feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=True)

    # Final plot
    importance_df.plot(kind='barh', x='feature', y='importance', legend=False, figsize=(10, 20))

    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()





def preprocess_and_resample_flag_rsv (df1, flag_value, input_test_size = 0.2, random_seed = 42, 
                             resampling_technique = None, ratio_maj_min = 0.8,
                             flag_column = 'season'):
    """
    Preprocesses the input data for the RSV phase 1 modelling stage, by applying feature 
    transformations and splitting the data into training and testing sets.

    Parameters:
    - df1 (DataFrame): The input DataFrame containing the data to be processed.
    - input_test_size (float): The proportion of the data to be used for testing. Defaults to 0.2.
    - random_seed (int): The random seed value for reproducibility. Defaults to 42.
    - flag_column (str): column that acts as a filter
    - flag_value (str / int / category): allows the preprocssing of data according to different categories within the data

    Returns:
    - X_train_transformed (ndarray): Transformed training features.
    - y_train (pd.Series): Training labels.
    - X_test_transformed (ndarray): Transformed testing features.
    - y_test (pd.Series): Testing labels.
    - preprocessor (ColumnTransformer): The preprocessor object used for feature transformations.
    """

    # Include the segmentation of the model according to the specific flag column we've chosen
    df1 = df1[df1[flag_column] == flag_value]

    # 0. To avoid data leakage, first thing to do is split data into train and test
    X = df1.drop(['RSV_test_result'], axis=1)
    y = df1['RSV_test_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= input_test_size, random_state=random_seed,
                                                        stratify= y)
    
    # 1. Select the features that are needed to be processed and how
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    # categorical_features.remove('RSV_test_result')
    categorical_mask = X_train.columns.isin(categorical_features)
    categorical_features.remove('calendar_year') # the reason behind this is that we will introduce manually the categories for calendar_year
    categorical_features.remove('sex') # the reason behind this is that we will introduce manually the categories for sex
    categorical_features.remove('race') # the reason behind this is that we will introduce manually the categories for race

    numeric_features_right = ['CCI', 'n_symptoms', 'prev_positive_rsv', 'previous_test_daydiff', 'n_immunodeficiencies']
    numeric_features_left = ['sine', 'cosine']

    # 2. Add resampling of the data
    print("Resampling method chosen:")

    if resampling_technique == "over":
        print("Oversampling")
        resampler = RandomOverSampler(sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique == "under":
        print("Undersampling")
        resampler = RandomUnderSampler(sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique == "smotenc":
        print("SMOTE-sampling")
        resampler = SMOTENC(categorical_features = categorical_mask,
                            sampling_strategy= ratio_maj_min,random_state=random_seed)

    elif resampling_technique is None:
        print("None")

    else:
        raise ValueError("Please, indicate a resampling technique. Accepted values are ['over', 'under', 'smotenc', None]")
    
    # 2.1 Fit the resampler to the data
    if resampling_technique is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)


    # 3. Define transformers for every feature type and build the preprocessor
    # 3.1 Categorical features first
    categorical_transformer = OneHotEncoder(drop = 'first')
    calendar_year_transformer = OneHotEncoder(categories= [sorted(list(df1['calendar_year'].unique()))] , drop = 'first')
    sex_transformer = OneHotEncoder(categories= [['Unknown', 'F', 'M']] , drop = 'first')
    race_transformer = OneHotEncoder(categories= [['Unknown', 'Asian', 'Black', 'Native American', 'Pacific', 'White']] , drop = 'first')

    # 3.2. Numeric features second
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
            ('cal_year',calendar_year_transformer, ['calendar_year']),
            ('se', sex_transformer, ['sex']),
            ('rac', race_transformer, ['race']),
            ('num_right', right_transformer, numeric_features_right),
            ('num_left', left_transformer, numeric_features_left)
        ])

    # 4. Finally, transform the data

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test, preprocessor



def train_model_flag_rsv(model, param_grid, target_scorer, n_cv_folds,
                    X_train, y_train, flag_value):
    """
    Trains a model for the RSV phase 1 modelling stage using grid search and cross-validation.

    Parameters:
    - model (object): The model object or estimator to be trained.
    - param_grid (dict): The dictionary of parameter grid for grid search.
    - target_scorer (str oor callable): The scoring metric for evaluating the model.
    - n_cv_folds (ind): The number of cross-validation folds.
    - X_train (nd-array): The training features (array-like or sparse matrix).
    - y_train (pd.series): The training labels (array-like).
    - flag_value (str / int / category): allows the preprocssing of data according to different categories within the data


    Returns:
    - grid_search (GridSearchCV): The trained grid search object.
    """
    print(f'Training model for flag == {flag_value} ... {model}')

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=target_scorer, cv=n_cv_folds)


    grid_search.fit(X_train, y_train)

    print('Best training parameters: ', grid_search.best_params_)
    print('Best training f1-score: ', grid_search.best_score_)

    return grid_search
