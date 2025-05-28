
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wilcoxon
from tqdm import tqdm  # Progress bar

# Load data
df_1_5T = pd.read_csv('/mnt/data/FRA/features/radiomics_features_1.5T.csv')
df_3T = pd.read_csv('/mnt/data/FRA/features/radiomics_features_3T.csv')

# Merge data
df = pd.merge(
    df_1_5T,
    df_3T,
    on=["subject_id", "Feature", "research_group"],
    suffixes=("_1.5T", "_3T")
)

# Extract feature type (assuming format: "original_{feature_type}_*")
df['feature_type'] = df['Feature'].str.split('_').str[1].str.lower()

# Ensure numeric columns
df['Value_1.5T'] = pd.to_numeric(df['Value_1.5T'], errors='coerce')
df['Value_3T'] = pd.to_numeric(df['Value_3T'], errors='coerce')
df = df.dropna(subset=['Value_1.5T', 'Value_3T'])

# ====================== Define analysis functions ======================
def kernel_ccc(x, y, gamma=0.5):
    """Compute nonlinear CCC based on RBF kernel"""
    x, y = np.array(x), np.array(y)
    Kx = rbf_kernel(x.reshape(-1, 1), gamma=gamma)
    Ky = rbf_kernel(y.reshape(-1, 1), gamma=gamma)
    n = len(x)
    H = np.eye(n) - np.ones((n, n))/n
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H
    return np.trace(Kx_c @ Ky_c) / np.sqrt(np.trace(Kx_c @ Kx_c) * np.trace(Ky_c @ Ky_c))

def bland_altman_plot(data, feat_name, ax=None, color='blue'):
    """Plot Bland-Altman diagram"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    mean = (data['Value_1.5T'] + data['Value_3T']) / 2
    diff = data['Value_1.5T'] - data['Value_3T']
    mean_diff = np.mean(diff)
    loa = 1.96 * np.std(diff)
    
    ax.scatter(mean, diff, alpha=0.5, color=color)
    ax.axhline(mean_diff, color='red', linestyle='--')
    ax.axhline(mean_diff + loa, color='green', linestyle=':')
    ax.axhline(mean_diff - loa, color='green', linestyle=':')
    ax.set_xlabel('Mean of 1.5T and 3T')
    ax.set_ylabel('Difference (1.5T - 3T)')
    ax.set_title(f'Bland-Altman: {feat_name}')
    return {
        'mean_diff': mean_diff,
        'upper_loa': mean_diff + loa,
        'lower_loa': mean_diff - loa,
        'points': list(zip(mean, diff))
    }

# ====================== Analyze by feature type ======================
results = []
feature_types = df[~df['feature_type'].isin(['image-original', 'mask-original'])]['feature_type'].unique()
df = df[~df['feature_type'].isin(['image-original', 'mask-original'])]

for ftype in tqdm(feature_types, desc='Analyzing feature types'):
    # Get all features of the current type
    features = df[df['feature_type'] == ftype]['Feature'].unique()
    
    for feat in tqdm(features, desc=f'Features ({ftype})', leave=False):
        sub_df = df[df['Feature'] == feat]
        
        # Compute kernel CCC (nonlinear concordance)
        try:
            kccc = kernel_ccc(sub_df['Value_1.5T'], sub_df['Value_3T'])
        except:
            kccc = np.nan
        
        # Perform Wilcoxon test (significance of difference)
        try:
            _, wpval = wilcoxon(sub_df['Value_1.5T'], sub_df['Value_3T'])
        except:
            wpval = np.nan
        
        # Bland-Altman analysis
        ba_stats = bland_altman_plot(sub_df, feat)
        
        # Save results
        results.append({
            'feature_type': ftype,
            'feature': feat,
            'kernel_ccc': kccc,
            'wilcoxon_p': wpval,
            'ba_mean_diff': ba_stats['mean_diff'],
            'ba_upper_loa': ba_stats['upper_loa'],
            'ba_lower_loa': ba_stats['lower_loa'],
            'n_samples': len(sub_df)
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# ====================== Visualization of results ======================
# 1. Kernel CCC distribution by feature type
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='feature_type', y='kernel_ccc', showfliers=False)
plt.axhline(0.9, color='red', linestyle='--', label='Excellent (CCC>0.9)')
plt.axhline(0.6, color='orange', linestyle=':', label='Acceptable (CCC>0.6)')
plt.xlabel('Feature Types', fontsize=14, fontweight='bold')
plt.ylabel('NCCC Value', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', rotation=45)
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('1.png', bbox_inches='tight', dpi=300)

# 2. Bland-Altman bias by feature type
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='feature_type', y='ba_mean_diff', showfliers=False)
plt.axhline(0, color='black', linestyle='-')
plt.title('Mean Difference by Feature Type (1.5T - 3T)')
plt.ylabel('Bias')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('2.png')

# ====================== Save results ======================
# Save detailed results
results_df.to_csv('feature_agreement_results.csv', index=False)

# Summary by feature type
summary_df = results_df.groupby('feature_type').agg({
    'kernel_ccc': ['mean', 'median', lambda x: sum(x > 0.8)/len(x)],
    'ba_mean_diff': ['mean', 'median', 'std'],
    'feature': 'count'
}).reset_index()
summary_df.columns = ['feature_type', 'mean_ccc', 'median_ccc', 'prop_ccc_good', 
                     'mean_bias', 'median_bias', 'std_bias', 'n_features']
summary_df.to_csv('feature_type_summary.csv', index=False)

print("Analysis complete! Results saved to CSV files.")
print("\nSummary by feature type:")
print(summary_df)

