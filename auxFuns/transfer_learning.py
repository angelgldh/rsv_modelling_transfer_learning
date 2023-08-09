import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, entropy
import matplotlib.pyplot as plt


def plot_histograms(feature_values_source, feature_values_target, feature_name):
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

def analyze_categorical_feature(source_data, target_data, feature_name):

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


def analyze_feature(source_data, target_data, feature_name, categorical=False):
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
