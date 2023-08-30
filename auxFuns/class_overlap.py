import pandas as pd
import tqdm as tqdm
import numpy as np
from scipy.spatial.distance import euclidean, hamming
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from auxFuns.modelling import train_model_rsv, find_optimal_moving_threshold, calculate_performance_metrics_rsv
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import BallTree
import scipy.stats

def plot_3FMDA_planes(df, hue_target, palette = None, main_title = '', s_size = 1):
    f, axes = plt.subplots(1, 3, figsize=(12, 6))

    sns.scatterplot(data=df, x=0, y=1, hue=hue_target, ax=axes[0], s = s_size, palette = palette)
    axes[0].set_title('Component 1 vs Component 2')

    sns.scatterplot(data=df, x=0, y=2, hue=hue_target, ax=axes[1], s = s_size, palette = palette)
    axes[1].set_title('Component 1 vs Component 3')

    sns.scatterplot(data=df, x=1, y=2, hue=hue_target, ax=axes[2], s = s_size, palette = palette)
    axes[2].set_title('Component 2 vs Component 3')

    f.suptitle(main_title, y=0.98)
    plt.subplots_adjust(top=0.8)

    plt.tight_layout()

def plot_5FMDA_planes(df, hue_target, palette = None, main_title = '', s_size = 1, figsize = (14,24)):
    f, axes = plt.subplots(5, 2, figsize=figsize)

    # Two-dimensional plots
    sns.scatterplot(data=df, x=0, y=1, hue=hue_target, ax=axes[0, 0], s = s_size, palette=palette)
    axes[0, 0].set_title('Component 1 vs Component 2')

    sns.scatterplot(data=df, x=0, y=2, hue=hue_target, ax=axes[0, 1], s = s_size, palette=palette)
    axes[0, 1].set_title('Component 1 vs Component 3')

    sns.scatterplot(data=df, x=0, y=3, hue=hue_target, ax=axes[1, 0], s = s_size, palette=palette)
    axes[1, 0].set_title('Component 1 vs Component 4')

    sns.scatterplot(data=df, x=0, y=4, hue=hue_target, ax=axes[1, 1], s = s_size, palette=palette)
    axes[1, 1].set_title('Component 1 vs Component 5')

    sns.scatterplot(data=df, x=1, y=2, hue=hue_target, ax=axes[2, 0], s = s_size, palette=palette)
    axes[2, 0].set_title('Component 2 vs Component 3')

    sns.scatterplot(data=df, x=1, y=3, hue=hue_target, ax=axes[2, 1], s = s_size, palette=palette)
    axes[2, 1].set_title('Component 2 vs Component 4')

    sns.scatterplot(data=df, x=1, y=4, hue=hue_target, ax=axes[3, 0], s = s_size, palette=palette)
    axes[3, 0].set_title('Component 2 vs Component 5')

    sns.scatterplot(data=df, x=2, y=3, hue=hue_target, ax=axes[3, 1], s = s_size, palette=palette)
    axes[3, 1].set_title('Component 3 vs Component 4')

    sns.scatterplot(data=df, x=2, y=4, hue=hue_target, ax=axes[4, 0], s = s_size, palette=palette)
    axes[4, 0].set_title('Component 3 vs Component 5')

    sns.scatterplot(data=df, x=3, y=4, hue=hue_target, ax=axes[4, 1], s = s_size, palette=palette)
    axes[4, 1].set_title('Component 4 vs Component 5')
    
    f.suptitle(main_title, y=0.98)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()


def find_non_overlap(df_a, df_b, distance_func):
    ZA = pd.DataFrame()
    ZB = pd.DataFrame()
    
    for i, row_a in tqdm(df_a.iterrows(), total=df_a.shape[0], desc='Processing df_a'):
        h_AB = np.mean([distance_func(row_a.values, row_b.values) for _, row_b in df_b.iterrows()])
        max_distance_a = max([distance_func(row_a.values, row_b.values) for _, row_b in df_b.iterrows()])
        
        if abs(row_a.values.max() - max_distance_a) > h_AB:
            ZA = pd.concat([ZA, pd.DataFrame(row_a).transpose()])

    for i, row_b in tqdm(df_b.iterrows(), total=df_b.shape[0], desc='Processing df_b'):
        h_BA = np.mean([distance_func(row_b.values, row_a.values) for _, row_a in df_a.iterrows()])
        max_distance_b = max([distance_func(row_b.values, row_a.values) for _, row_a in df_a.iterrows()])
        
        if abs(row_b.values.max() - max_distance_b) > h_BA:
            ZB = pd.concat([ZB, pd.DataFrame(row_b).transpose()])

    Z = pd.concat([ZA, ZB])
    X = pd.concat([df_a, df_b])
    U = X.loc[X.index.difference(Z.index)]
    
    return Z, U

def find_non_overlap_fast(df_a, df_b, distance_func):
    ZA_rows = []
    ZB_rows = []

    # Pre-compute distance matrices
    dist_matrix_ab = pairwise_distances(df_a.values, df_b.values, metric=distance_func)
    dist_matrix_ba = pairwise_distances(df_b.values, df_a.values, metric=distance_func)

    # Processing df_a
    for i in tqdm(range(df_a.shape[0]), desc='Processing df_a'):
        h_AB = dist_matrix_ab[i].mean()
        max_distance_a = dist_matrix_ab[i].max()
        if abs(df_a.values[i].max() - max_distance_a) > h_AB:
            ZA_rows.append(df_a.iloc[i])

    # Processing df_b
    for i in tqdm(range(df_b.shape[0]), desc='Processing df_b'):
        h_BA = dist_matrix_ba[i].mean()
        max_distance_b = dist_matrix_ba[i].max()
        if abs(df_b.values[i].max() - max_distance_b) > h_BA:
            ZB_rows.append(df_b.iloc[i])

    ZA = pd.DataFrame(ZA_rows)
    ZB = pd.DataFrame(ZB_rows)

    Z = pd.concat([ZA, ZB])
    X = pd.concat([df_a, df_b])
    U = X.loc[X.index.difference(Z.index)]
    
    return Z, U

def pred_custom(model1, X_test, optimal_threshold):
    y_probs = model1.predict_proba(X_test)[:, 1]
    pred =  ["Positive" if p > optimal_threshold else "Negative"  for p in y_probs]
    return y_probs, pred

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive']).ravel()
    return tn / (tn+fp)

def negative_predictive_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive']).ravel()
    return tn / (tn+fn)

def build_and_evaluate_2overlapping_models(df1, same_class_neighbors, test_size = 0.2, random_seed = 42,
                                           model_class_non_overlapping = RandomForestClassifier(),
                                           param_grid_non_overlapping = {'n_estimators': [7, 14],'max_depth': [10, 20],'min_samples_split': [5, 10],'min_samples_leaf': [1, 4]},
                                           cost_sensitive_non_overlapping = True, weight_dict_non_overlapping = {'Negative': 1, 'Positive': 15},
                                           model_class_overlapping = RandomForestClassifier(),
                                           param_grid_overlapping = {'n_estimators': [7, 14],'max_depth': [10, 20],'min_samples_split': [5, 10],'min_samples_leaf': [1, 4]},
                                           cost_sensitive_overlapping = True, weight_dict_overlapping = {'Negative': 1, 'Positive': 15},

                                           ):
    # ----------------------------------
    # Model for NON-overlapping region

    print('----------------')
    print('Building non-overlapping model ...')

    df_non_overlapping = df1.loc[ same_class_neighbors == True,:]

    X = df_non_overlapping.drop(['RSV_test_result'], axis=1)
    y = df_non_overlapping['RSV_test_result']
    X_train, X_test_nonOverlapping, y_train, y_test_nonOverlapping = train_test_split(X, y, test_size= test_size, random_state=random_seed,
                                                        stratify= y)

    X_train = pd.get_dummies(X_train)
    X_test_nonOverlapping = pd.get_dummies (X_test_nonOverlapping)

    if cost_sensitive_non_overlapping:
        model_class_non_overlapping.set_params(class_weight=weight_dict_non_overlapping, random_state=random_seed)
    else:
        model_class_non_overlapping.set_params(class_weight=None, random_state=random_seed)


    target_scorer = make_scorer(f1_score, average='binary', pos_label = 'Positive')
    n_cv_folds = 5

    model1_nonOverlapping = train_model_rsv(model = model_class_non_overlapping, param_grid = param_grid_non_overlapping, target_scorer = target_scorer, n_cv_folds = n_cv_folds,
                        X_train = X_train, y_train = y_train)
    
    # Evaluation of the non-overlapping region
    print('\n----------------')
    print('Performance metrics of non-overlapping model ...')
    optimal_threshold_nonOverlapping = find_optimal_moving_threshold(trained_model = model1_nonOverlapping, X_test = X_test_nonOverlapping, y_test = y_test_nonOverlapping)

    __,__,__,__,__,__,f1,__ = calculate_performance_metrics_rsv(trained_model = model1_nonOverlapping, X_test = X_test_nonOverlapping, y_test = y_test_nonOverlapping,
                                                         threshold = optimal_threshold_nonOverlapping, 
                                                         print_roc = False, print_pr = False)
    
    # ----------------------------------
    # Model for Overlapping region
    print('----------------')
    print('Building (yes) overlapping model ...')
    
    df_overlapping = df1.loc[ same_class_neighbors == False,:]

    X = df_overlapping.drop(['RSV_test_result'], axis=1)
    y = df_overlapping['RSV_test_result']
    X_train, X_test_Overlapping, y_train, y_test_Overlapping = train_test_split(X, y, test_size= test_size, random_state=random_seed,
                                                            stratify= y)

    X_train = pd.get_dummies(X_train)
    X_test_Overlapping = pd.get_dummies (X_test_Overlapping)

    if cost_sensitive_overlapping:
        model_class_overlapping.set_params(class_weight=weight_dict_overlapping, random_state=random_seed)
    else:
        model_class_overlapping.set_params(class_weight=None, random_state=random_seed)


    target_scorer = make_scorer(f1_score, average='binary', pos_label = 'Positive')
    n_cv_folds = 5

    model1_Overlapping = train_model_rsv(model = model_class_overlapping, param_grid = param_grid_overlapping, target_scorer = target_scorer, n_cv_folds = n_cv_folds,
                        X_train = X_train, y_train = y_train)
    
    # Evaluation of the overlapping region
    print('\n----------------')
    print('Performance metrics of (yes) overlapping model ...')
    optimal_threshold_Overlapping = find_optimal_moving_threshold(trained_model = model1_Overlapping, X_test = X_test_Overlapping, y_test = y_test_Overlapping)

    __,__,__,__,__,__,f1,__ = calculate_performance_metrics_rsv(trained_model = model1_Overlapping, X_test = X_test_Overlapping, y_test = y_test_Overlapping,
                                                         threshold = optimal_threshold_Overlapping, 
                                                         print_roc = False, print_pr = False)

    # Evaluation of the aggregated model

    y_probs_nonOverlapping, pred_nonOverlapping = pred_custom(model1 = model1_nonOverlapping, X_test = X_test_nonOverlapping, optimal_threshold = optimal_threshold_nonOverlapping)
    y_probs_overlapping, pred_overlapping = pred_custom(model1 = model1_Overlapping, X_test = X_test_Overlapping, optimal_threshold = optimal_threshold_Overlapping)

    pred = pred_nonOverlapping + pred_overlapping
    y_probs = np.concatenate([y_probs_nonOverlapping, y_probs_overlapping])
    true = pd.concat([y_test_nonOverlapping, y_test_Overlapping], axis = 0)

    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, pos_label = 'Positive')
    recall = recall_score(true, pred, pos_label = 'Positive')
    f1 = f1_score(true, pred, pos_label = 'Positive')
    roc_auc = roc_auc_score(true, y_probs)
    spec = specificity(true, pred)
    npv = negative_predictive_value(true, pred)
    precision_curve, recall_curve, _ = precision_recall_curve(true, y_probs, pos_label = 'Positive')
    pr_auc = auc(recall_curve, precision_curve)

    print('\n-------')
    print('Performance metrics of the aggregated model')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall/Sensitivity: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Specificity: {spec}")
    print(f"Negative Predictive Value: {npv}")
    print(f"Precision-Recall AUC: {pr_auc}")


    
    return model1_nonOverlapping, model1_Overlapping

def rename_key(d, old_key, new_key):
    if old_key in d:
        d[new_key] = d.pop(old_key)
    return d

def only_build_2overlapping_models(X, labels, same_class_neighbors, random_seed = 42,
                                           model_class_non_overlapping = RandomForestClassifier(),
                                           param_grid_non_overlapping = {'n_estimators': [7, 14],'max_depth': [10, 20],'min_samples_split': [5, 10],'min_samples_leaf': [1, 4]},
                                           cost_sensitive_non_overlapping = True, weight_dict_non_overlapping = {'Negative': 1, 'Positive': 15},
                                           model_class_overlapping = RandomForestClassifier(),
                                           param_grid_overlapping = {'n_estimators': [7, 14],'max_depth': [10, 20],'min_samples_split': [5, 10],'min_samples_leaf': [1, 4]},
                                           cost_sensitive_overlapping = True, weight_dict_overlapping = {'Negative': 1, 'Positive': 15},
                                           pos_label = 'Positive', neg_label = 'Negative', label_name = 'RSV_test_result',
                                           sample_weights_Overlapping = None, sample_weights_nonOverlapping = None
                                           ):
    rename_key(d = weight_dict_non_overlapping, old_key = 'Positive', new_key = pos_label)
    rename_key(d = weight_dict_non_overlapping, old_key = 'Negative', new_key = neg_label)

    rename_key(d = weight_dict_overlapping, old_key = 'Positive', new_key = pos_label)
    rename_key(d = weight_dict_overlapping, old_key = 'Negative', new_key = neg_label)

    # ----------------------------------
    # Model for NON-overlapping region
    print('----------------')
    print('Building non-overlapping model ...')

    X = X.reset_index(drop = True)
    labels = labels.reset_index(drop = True)
    df1 = pd.concat([X, labels], axis = 1)

    df_non_overlapping = df1.loc[ same_class_neighbors == True,:]

    X_non_overlapping = df_non_overlapping.drop([label_name], axis=1)
    y_non_overlapping = df_non_overlapping[label_name]

    X_non_overlapping = pd.get_dummies(X_non_overlapping)

    if cost_sensitive_non_overlapping:
        model_class_non_overlapping.set_params(class_weight=weight_dict_non_overlapping, random_state=random_seed)
    else:
        model_class_non_overlapping.set_params(class_weight=None, random_state=random_seed)


    target_scorer = make_scorer(f1_score, average='binary', pos_label = pos_label)
    n_cv_folds = 5

    model1_nonOverlapping = train_model_rsv(model = model_class_non_overlapping, param_grid = param_grid_non_overlapping, target_scorer = target_scorer, n_cv_folds = n_cv_folds,
                        X_train = X_non_overlapping, y_train = y_non_overlapping, sample_weights=sample_weights_nonOverlapping)
    
    # # Evaluation of the non-overlapping region
    # print('\n----------------')
    # print('Performance metrics of non-overlapping model ...')
    # optimal_threshold_nonOverlapping = find_optimal_moving_threshold(model = model1_nonOverlapping, X_test = X_test_nonOverlapping, y_test = y_test_nonOverlapping)

    # __,__,__,__,__,__,f1,__ = calculate_performance_metrics_rsv(trained_model = model1_nonOverlapping, X_test = X_test_nonOverlapping, y_test = y_test_nonOverlapping,
    #                                                      threshold = optimal_threshold_nonOverlapping, 
    #                                                      print_roc = False, print_pr = False)
    
    # ----------------------------------
    # Model for Overlapping region
    print('----------------')
    print('Building (yes) overlapping model ...')

    if (same_class_neighbors == False).sum() == 0:
        print('Warning! No overlapping classes')
        model1_Overlapping = None
    else:   
        df_overlapping = df1.loc[ same_class_neighbors == False,:]

        X_overlapping = df_overlapping.drop([label_name], axis=1)
        y_overlapping = df_overlapping[label_name]

        X_overlapping = pd.get_dummies(X_overlapping)

        if cost_sensitive_overlapping:
            model_class_overlapping.set_params(class_weight=weight_dict_overlapping, random_state=random_seed)
        else:
            model_class_overlapping.set_params(class_weight=None, random_state=random_seed)


        target_scorer = make_scorer(f1_score, average='binary', pos_label = 'Positive')
        n_cv_folds = 5

        model1_Overlapping = train_model_rsv(model = model_class_overlapping, param_grid = param_grid_overlapping, target_scorer = target_scorer, n_cv_folds = n_cv_folds,
                            X_train = X_overlapping, y_train = y_overlapping, sample_weights= sample_weights_Overlapping)
    
    
    return model1_nonOverlapping, model1_Overlapping



def distance_matrix_to_percentile(matrix):
    """
    Convert a distance matrix to its corresponding percentile matrix.
    """
    # Flatten the matrix and compute the rank for each distance
    flat_matrix = matrix.flatten()
    ranks = scipy.stats.rankdata(flat_matrix)
    
    # Convert ranks to percentiles
    percentiles = 100.0 * ranks / (len(flat_matrix) - 1)
    
    # Reshape the flattened percentiles back to the original shape
    percentile_matrix = percentiles.reshape(matrix.shape)
    
    return percentile_matrix


def calculate_same_neighbours_and_N1(X ,y, n_neighbours = 5):

    # Step 1: Ball tree 
    print('Build-up of the ball tree ...')

    tree2 = BallTree(X)

    # Step 2: tree-guided closest neighbour
    # The closest point to every instance is itself, so k = 2 returns the point itself + the closest neighbour
    # let us take 5 (4 neighbours + 1 itself) closest neighbours to help in the prediction of overlapping instances 
    print(f'Finding closest {n_neighbours - 1} neighbour(s) in the ball tree for every instance ...')
    dist2, ind2 = tree2.query(X, k=n_neighbours)

    # Step 3: same class neighbours
    print('Determination of instances with same-class neighbours ...')
    labels = y.to_numpy()
    # same_class_neighbours2 = labels[ind2[:, 0]] == labels[ind2[:, 1]]
    # N1_2 = 1 - np.mean(same_class_neighbours2)
    # print(f'N1 metric: {N1_2}')

    # Step 4: add a higher number of neighbours to evaluate a varying overlapping
    same_class_neighbours_dict = {}
    N1_dict = {}

    for k in range(1, n_neighbours):
        # For each k, we'll check for same class neighbors across all instances
        same_class_neighbours2 = np.array([all(labels[i] == label for label in labels[ind2[j, 1:k+1]]) for j, i in enumerate(ind2[:, 0])])
        N1_metric = 1 - np.mean(same_class_neighbours2)

        same_class_neighbours_dict[k] = same_class_neighbours2
        N1_dict[k] = N1_metric 

        print(f'N1 metric for {k} neighbours : {N1_metric}')

    print('Done!')

    return same_class_neighbours_dict, N1_dict, dist2, ind2


def percentile_distances_df_buildup(percentile_matrix):
    percentile_distance_df = pd.DataFrame(percentile_matrix, columns = ['percentile_distance_to_neighbour_n_' + str(ii + 1) for ii in range(percentile_matrix.shape[1])])
    percentile_distance_df['mean_percentile_distance'] = np.mean(percentile_matrix, axis = 1)
    percentile_distance_df['min_percentile_distance'] = np.min(percentile_matrix, axis = 1)
    percentile_distance_df['max_percentile_distance'] = np.max(percentile_matrix, axis = 1)
    percentile_distance_df['std_percentile_distance'] = np.std(percentile_matrix, axis = 1)

    return percentile_distance_df