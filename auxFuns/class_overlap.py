import pandas as pd
import tqdm as tqdm
import numpy as np
from scipy.spatial.distance import euclidean, hamming
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

def plot_3FMDA_planes(df, hue_target, palette = None, main_title = ''):
    f, axes = plt.subplots(1, 3, figsize=(12, 6))

    sns.scatterplot(data=df, x=0, y=1, hue=hue_target, ax=axes[0], s = 1, palette = palette)
    axes[0].set_title('Component 1 vs Component 2')

    sns.scatterplot(data=df, x=0, y=2, hue=hue_target, ax=axes[1], s = 1, palette = palette)
    axes[1].set_title('Component 1 vs Component 3')

    sns.scatterplot(data=df, x=1, y=2, hue=hue_target, ax=axes[2], s = 1, palette = palette)
    axes[2].set_title('Component 2 vs Component 3')

    f.suptitle(main_title, y=0.98)
    plt.subplots_adjust(top=0.8)

    plt.tight_layout()

def plot_5FMDA_planes(df, hue_target, palette = None, main_title = '', s_size = 1):
    f, axes = plt.subplots(5, 2, figsize=(14, 24))

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