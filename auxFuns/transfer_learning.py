import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score


def plot_histograms_v0(feature_values_source, feature_values_target, feature_name):
    plt.figure(figsize=(5, 3))
    source = feature_values_source
    target = feature_values_target
    
    # Get histogram data
    bins = np.linspace(min(source.min(), target.min()), max(source.max(), target.max()), 50)
    source_hist, _ = np.histogram(source, bins=bins, density=True)
    target_hist, _ = np.histogram(target, bins=bins, density=True)
    
    # Compute overlap
    overlap_hist = np.minimum(source_hist, target_hist)
    
    # Plotting
    plt.hist(bins[:-1], bins, weights=source_hist, alpha=0.3, color='blue', label='Source')
    plt.hist(bins[:-1], bins, weights=target_hist, alpha=0.3, color='red', label='Target')
    plt.hist(bins[:-1], bins, weights=overlap_hist, alpha=0.6, color='green', label='Overlap')

    plt.title(f"Histogram for {feature_name}")
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Calculate histogram intersection as a fraction
    bin_width = bins[1] - bins[0]
    histogram_intersection = np.sum(overlap_hist * bin_width)
    overlap_area = np.sum(overlap_hist * bin_width)

    print(f'Overlap area: {overlap_area}')
    return overlap_area

def plot_histograms(ax, feature_values_source, feature_values_target, feature_name):
    source = feature_values_source
    target = feature_values_target
    
    # Get histogram data
    bins = np.linspace(min(source.min(), target.min()), max(source.max(), target.max()), 50)
    source_hist, _ = np.histogram(source, bins=bins, density=True)
    target_hist, _ = np.histogram(target, bins=bins, density=True)
    
    # Compute overlap
    overlap_hist = np.minimum(source_hist, target_hist)
    
    # Plotting
    ax.hist(bins[:-1], bins, weights=source_hist, alpha=0.3, color='blue', label='Source')
    ax.hist(bins[:-1], bins, weights=target_hist, alpha=0.3, color='red', label='Target')
    ax.hist(bins[:-1], bins, weights=overlap_hist, alpha=0.6, color='green', label='Overlap')

    ax.set_title(f"Histogram for {feature_name}")
    ax.set_ylabel('Density')
    ax.legend()

    # Calculate histogram intersection as a fraction
    bin_width = bins[1] - bins[0]
    overlap_area = np.sum(overlap_hist * bin_width)

    return overlap_area


def calculate_kl_divergence(p, q):
    """Compute KL divergence of two distributions."""
    # Ensure the distributions are normalized
    p = p/p.sum()
    q = q/q.sum()

    # Clip values to avoid zeros
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    
    return entropy(p, q)

def histogram_intersection(p, q):
    return np.sum(np.minimum(p, q))

def analyze_categorical_feature_v0(source_data, target_data, feature_name):

    source = source_data[feature_name]
    target = target_data[feature_name]
    categories = list(set(source.unique()) | set(target.unique()))
    
    # Convert categorical variables to a proportion representation
    source_counts = source.value_counts(normalize=True).reindex(categories, fill_value=0)
    target_counts = target.value_counts(normalize=True).reindex(categories, fill_value=0)
    
    # Plot
    ind = np.arange(len(categories)) 
    width = 0.3  
    plt.figure(figsize=(5, 3))
    plt.bar(ind - width/2, source_counts.values, width=width, label='Source', alpha=0.3, color='blue')
    plt.bar(ind + width/2, target_counts.values, width=width, label='Target', alpha=0.3, color='red')
    plt.xticks(ind, categories, rotation=45)  # Set category labels as x-ticks
    plt.title(f'Proportion plot for {feature_name}')
    plt.legend()
    plt.tight_layout()  
    plt.show()

    # Compute chi-squared test
    contingency_table = pd.crosstab(source, target, margins=False)
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Squared Statistic: {chi2_stat}, P-value: {p_val}")
    display(pd.crosstab(source, target, margins=True, margins_name="Total"))

    # Compute histogram intersection
    intersection = histogram_intersection(source_counts, target_counts)
    print(f"Histogram Intersection: {intersection}")


def domain_shift_analyze_feature_v0(source_data, target_data, feature_name, categorical=False):
    print(f"Analyzing feature: {feature_name}\n{'='*30}")
    
    if categorical:
        analyze_categorical_feature(source_data, target_data, feature_name)
    else:
        # Plot histogram
        plot_histograms(source_data[feature_name], target_data[feature_name], feature_name)
        
        # Compute KS Test
        D_statistic, p_value = ks_2samp(source_data[feature_name], target_data[feature_name])
        print(f"KS Statistic: {D_statistic}, P-value: {p_value}")
        
        # Compute KL Divergence (using histogram bins)
        source_hist, bin_edges = np.histogram(source_data[feature_name], bins=50, density=True)
        target_hist, _ = np.histogram(target_data[feature_name], bins=bin_edges, density=True)
        
        kl_div = calculate_kl_divergence(source_hist, target_hist)
        print(f"KL Divergence: {kl_div}")

        # Compute histogram intersection
        intersection = histogram_intersection(source_hist, target_hist)
        print(f"Histogram Intersection: {intersection}")
    print("\n")

def create_metrics_table(ax, metrics):
    col_labels = ['Metric', 'Value']
    ax.axis('off')
    ax.table(cellText=metrics, colLabels=col_labels, loc='center')


def analyze_categorical_feature(axs, source_counts, target_counts, feature_name, chi2_stat, p_val, intersection):

    ind = np.arange(len(source_counts.index))
    width = 0.3
    axs[0].bar(ind - width/2, source_counts.values, width=width, label='Source', alpha=0.3, color='blue')
    axs[0].bar(ind + width/2, target_counts.values, width=width, label='Target', alpha=0.3, color='red')
    axs[0].set_xticks(ind)
    axs[0].set_xticklabels(source_counts.index, rotation=45)
    axs[0].set_title(f'Proportion plot for {feature_name}')
    axs[0].legend()

     # Metric table
    metrics = [['Chi-Squared Statistic', chi2_stat], ['P-value', p_val], ['Histogram Intersection', intersection]]
    create_metrics_table(axs[1], metrics)


def domain_shift_analyze_feature(source_data, target_data, feature_name, categorical=False):
    print(f"Analyzing feature: {feature_name}\n{'='*30}")
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 3))
    
    if categorical:
        source = source_data[feature_name]
        target = target_data[feature_name]
        categories = list(set(source.unique()) | set(target.unique()))
        source_counts = source.value_counts(normalize=True).reindex(categories, fill_value=0)
        target_counts = target.value_counts(normalize=True).reindex(categories, fill_value=0)
        
        # Compute chi-squared test
        contingency_table = pd.crosstab(source, target, margins=False)
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        
        # Compute histogram intersection
        intersection = histogram_intersection(source_counts, target_counts)
        
        analyze_categorical_feature(axs, source_counts, target_counts, feature_name, chi2_stat, p_val, intersection)
    else:
        # Plot histogram in axs[0]
        overlap_area = plot_histograms(axs[0], source_data[feature_name], target_data[feature_name], feature_name)
        
        # Compute KS Test
        D_statistic, p_value = ks_2samp(source_data[feature_name], target_data[feature_name])
        
        # Compute KL Divergence (using histogram bins)
        source_hist, bin_edges = np.histogram(source_data[feature_name], bins=50, density=True)
        target_hist, _ = np.histogram(target_data[feature_name], bins=bin_edges, density=True)
        
        kl_div = calculate_kl_divergence(source_hist, target_hist)  # Assuming you have a function to calculate this
        
        # Compute histogram intersection
        intersection = histogram_intersection(source_hist, target_hist)  # Assuming you have a function to calculate this

        # Add metrics as a table to the right subplot
        col_labels = ['Metric', 'Value']
        row_labels = ['Overlap Area', 'KS Statistic', 'P-value', 'KL Divergence', 'Histogram Intersection']
        table_vals = [[metric, value] for metric, value in zip(row_labels, [overlap_area, D_statistic, p_value, kl_div, intersection])]
        
        create_metrics_table(axs[1], table_vals)  # You might need to define this function to plot the table
    
    plt.show()
    print("\n")
        
        
    plt.show()
    print("\n")


def wasserstein_balltree(source, target, n_neighbours=2):
    """
    Compute the Wasserstein-like distance between source and target data using BallTree method.
    Parameters:
    - source (pd.DataFrame): features (X) of the source population
    - target (pd.DataFrame): features (X) of the target population
    - n-neighbours (int, float): number of closest neighbours (n_neighbours - 1) to determine. The closest is always itself, so 2 ensures it detects the closest-and different-instance

    Return:
    - wasserstein_equivalent (float): Wasserstein distance-equivalent built with the ball tree. Gives a measure of 'how much effort' is needed to take every point in the one population to the other
    """
    # Step 1: Compute the BallTree for source and target
    tree_source = BallTree(source)
    tree_target = BallTree(target)

    # Step 2: For each point in source, find its closest point in target
    dist_to_target, _ = tree_source.query(target, k=n_neighbours)
    average_distance_to_target = np.mean(dist_to_target[:, 1])  # exclude the 0th column since it's distance to itself

    # Step 3: For each point in target, find its closest point in source
    dist_to_source, _ = tree_target.query(source, k=n_neighbours)
    average_distance_to_source = np.mean(dist_to_source[:, 1])  # exclude the 0th column since it's distance to itself

    # Compute the symmetric Wasserstein-like distance
    wasserstein_equivalent = (average_distance_to_target + average_distance_to_source) / 2.0

    return wasserstein_equivalent




def domain_shift_discriminative_analysis (DA_dictionary, X_modelling, labels, pos_label_target_domain = 2):
    """
    Function to perform discriminative analysis on data that might be subjected to domain shift.
    
    Parameters:
    - DA_dictionary (dict): A dictionary containing the models to be used and empty lists for metrics to be stored.
        Expected keys:
            'model': List of classifiers
            'y_probs': List to store probabilities
            'error_rate': List to store error rates
            'accuracy': List to store accuracy scores
            'precision': List to store precision scores
            'recall': List to store recall scores
            'f1_score': List to store f1 scores
            'roc_auc_score': List to store ROC AUC scores
    
    - X_modelling (array-like): The feature set to be used for modeling. Shape (n_samples, n_features)
    - labels (array-like): Ground truth labels. Shape (n_samples,)
    
    Returns:
    - DA_dictionary (dict): Updated dictionary with all calculated metrics.
    
    Notes:
    - The function assumes that the classifiers in the DA_dictionary are already initialized.
    - Assumes a binary classification problem.
    """
    for ii in range(len(DA_dictionary['model'])):
        clf = DA_dictionary['model'][ii]
        X_train, X_test, y_train, y_test = train_test_split(X_modelling, labels, test_size=0.7, stratify=labels)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_probs = clf.predict_proba(X_test)[:, 1]  #Probability of belonging to phase 2
        DA_dictionary['y_probs'][ii] = y_probs

        # Calculate and store various metrics
        acc = accuracy_score(y_test, y_pred)
        error = 1 - acc
        DA_dictionary['error_rate'][ii] = error
        
        DA_dictionary['accuracy'][ii] = acc
        DA_dictionary['precision'][ii] = precision_score(y_test, y_pred, pos_label = pos_label_target_domain)
        DA_dictionary['recall'][ii] = recall_score(y_test, y_pred, pos_label = pos_label_target_domain)
        DA_dictionary['f1_score'][ii] = f1_score(y_test, y_pred, pos_label = pos_label_target_domain)
        DA_dictionary['roc_auc_score'][ii] = roc_auc_score(y_test, y_probs)

        print(f"\nMetrics for {type(clf).__name__}:")
        print(f"Error rate: {error:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {DA_dictionary['precision'][ii]:.4f}")
        print(f"Recall: {DA_dictionary['recall'][ii]:.4f}")
        print(f"F1 Score: {DA_dictionary['f1_score'][ii]:.4f}")
        print(f"ROC AUC Score: {DA_dictionary['roc_auc_score'][ii]:.4f}")

    return DA_dictionary

def domain_shift_plot_histogram_DA(ax, y_probs, num_samples_source, num_samples_target, model_name='Model'):
    """
    Plot histogram of predicted probabilities of belonging to a target domain
    and special bins for number of source and target domain samples.
    
    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib subplot object.
        y_probs (array-like): Probabilities of the positive class.
        num_samples_source (int): Number of samples in the source domain.
        num_samples_target (int): Number of samples in the target domain.
        model_name (str): Name of the model (for labeling purposes).
    """
    ax.hist(y_probs, bins=50, alpha=0.4, label=f"{model_name}")
    # Imporant: do not include density = True in this hist
    
    # Add special bins for source and target domain samples
    width_1 = 0.03
    width_2 = 0.03
    ax.bar((width_1/2), num_samples_source, color='none', edgecolor='black', width=width_1)
    ax.bar((1 - (width_2/2)), num_samples_target, color='none', edgecolor='black', width=width_2)
    
    # Add arrows with annotations
    ax.annotate(f'Number of samples source', xy=(0, num_samples_source), xytext=(0.2, num_samples_source + 0.1),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax.annotate(f'Number of samples target', xy=(1, num_samples_target), xytext=(0.7, num_samples_target + 0.1),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    ax.set_xlabel('Predicted Probability of belonging to target domain')
    ax.set_ylabel('Count')
    ax.legend(loc='upper left')
    ax.grid(True)

def domain_shift_create_summary_table_labels(source_data, target_data, label_column, patient_id_column):
    """
    Create a summary table comparing the source and target datasets on various metrics.
    Specific to the quantification of domain shift in the RSV transfer learning project.
    
    Parameters:
    -----------
    source_data : pd.DataFrame
        The source data containing label and patient information.
        
    target_data : pd.DataFrame
        The target data containing label and patient information.
        
    label_column : str
        The column name in both dataframes that contains the labels (Positive or Negative).
        
    patient_id_column : str
        The column name in both dataframes that contains the unique patient identifiers.
        
    Returns:
    --------
    pd.DataFrame.style
        A styled DataFrame containing the summary information. 
    
    The returned DataFrame has the following metrics:
    - '# Positives': The count of positive samples.
    - '# Negatives': The count of negative samples.
    - 'Imbalance ratio': The ratio of negative samples to positive samples.
    - '# Unique Patients': The count of unique patients.
    
    Notes:
    ------
    - Assumes that the positive and negative classes in `label_column` are represented as 1 and 0.
    - Assumes that the value_counts function will return the labels as 'Positive' and 'Negative'.
    """
    summary_data = {
        'Source': [
            source_data[label_column].value_counts().get(1, 0),
            source_data[label_column].value_counts().get(0, 0),
            source_data[label_column].value_counts()['Negative']/source_data[label_column].value_counts()['Positive'],
            source_data[patient_id_column].nunique()
        ],
        'Target (labelled subset)': [
            target_data[label_column].value_counts().get(1, 0),
            target_data[label_column].value_counts().get(0, 0),
            target_data[label_column].value_counts()['Negative']/target_data[label_column].value_counts()['Positive'],
            target_data[patient_id_column].nunique()
        ]
    }
    summary_df = pd.DataFrame(summary_data, index=['# Positives', '# Negatives', 'Imbalance ratio','# Unique Patients'])

    # Pre-format the DataFrame
    for col in summary_df.columns:
        for idx in summary_df.index:
            if idx == 'Imbalance ratio':
                summary_df.loc[idx, col] = "{:,.3f}".format(summary_df.loc[idx, col])
            else:
                summary_df.loc[idx, col] = "{:,.0f}".format(summary_df.loc[idx, col])
                
    return summary_df.style

def label_shift_plot_combined_chart(labels_source, labels_target, source_data, target_data, label_column, patient_id_column):
    """
    Plot a combined chart to visualize a label shift between source and target domain.
    
    Parameters:
    - labels_source (pd.Series): Series of labels from the source dataset.
    - labels_target (pd.Series): Series of labels from the target dataset.
    - source_data (pd.DataFrame): Source data frame containing label_column and patient_id_column.
    - target_data (pd.DataFrame): Target data frame containing label_column and patient_id_column.
    - label_column (str): The column name representing the label in source_data and target_data.
    - patient_id_column (str): The column name representing the unique patient IDs.
    
    Output:
    A matplotlib figure with three subplots:
    1. A bar chart representing the absolute counts of positive and negative samples in the source and target datasets.
    2. A pie chart representing the proportion of positive to negative samples in the source dataset.
    3. A pie chart representing the proportion of positive to negative samples in the target dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    
    # 1. Bar Chart - absolute number of 
    df = pd.concat([
        labels_source.value_counts().rename('Source'),
        labels_target.value_counts().rename('Target (labelled subset)')
    ], axis=1).fillna(0)
    df_t = df.transpose()
    light_red = (0.9, 0.4, 0.4) # RGB for light red
    light_green = (0.4, 0.9, 0.4) # RGB for light green
    light_blue = (0.4, 0.4, 0.9) # RGB for light blue

    ax1 = df_t.plot(kind='bar', ax=axes[0], color=[light_blue, light_red], logy=True)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    for p in ax1.patches:
        ax1.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
    axes[0].set_title('RSV test results in Source and Target Domains')
    axes[0].set_ylabel('Count (log scale)')
    
    # 2. Pie Chart for Source
    # Pie charts indicate the percentage of positivity per unique patients
    source_unique = source_data.drop_duplicates(subset=[patient_id_column])
    source_unique[label_column].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', colors = [light_blue, light_red])
    axes[1].set_title('Source: RSV positivity per unique patient')
    axes[1].set_ylabel('')
    
    # 3. Pie Chart for Target
    target_unique = target_data.drop_duplicates(subset=[patient_id_column])
    target_unique[label_column].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', colors = [light_blue, light_red])
    axes[2].set_title('Target: RSV positivity per unique patient')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.show()


