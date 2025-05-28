import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*")



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, score_func=mutual_info_classif):
        self.k = k
        self.score_func = score_func

    def fit(self, X, y):
        self.selector = SelectKBest(score_func=self.score_func, k=self.k)
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()]
        return self

    def transform(self, X):
        return self.selector.transform(X)


# 1. Load and reshape data
def load_and_prepare(scanner):
    """Load and reshape the data for a specific scanner."""
    df = pd.read_csv(f'/mnt/data/FRA/features/selected_features_{scanner}.csv')

    # Convert to wide format
    df_wide = df.pivot_table(
        index=['subject_id', 'research_group'],
        columns='Feature',
        values='Value',
        aggfunc='first'
    ).reset_index()

    return df_wide


# 2. Evaluation with cross-validation, feature selection, and class balancing
def evaluate_model(scanner, task='MCI_vs_NC', n_splits=5, random_state=42):
    """Evaluate classification performance using cross-validation with feature selection and ADASYN."""
    df_wide = load_and_prepare(scanner)

    # Define binary classification tasks
    if task == 'MCI_vs_NC':
        df_wide = df_wide[df_wide['research_group'].isin(['MCI', 'CN'])]
    elif task == 'MCI_vs_AD':
        df_wide = df_wide[df_wide['research_group'].isin(['MCI', 'AD'])]

    X = df_wide.drop(columns=['subject_id', 'research_group'])
    y = (df_wide['research_group'] == 'MCI').astype(int)

    print(f"\nTask: {task}, Scanner: {scanner} — Class distribution: {np.bincount(y)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []

    fig, ax = plt.subplots(figsize=(8, 6))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train/Validation Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)

        # Define pipeline
        pipeline = Pipeline([
            ('feature_selection', FeatureSelector(k=10, score_func=mutual_info_classif)),
            ('resampler', ADASYN(random_state=random_state)),
            ('classifier', RandomForestClassifier(
                n_estimators=350,
                max_depth=10,
                class_weight='balanced',
                random_state=random_state
            ))
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate on validation
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Evaluate on test
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        auc_scores.append(test_auc)

        # Plot ROC curve
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, alpha=0.3, lw=1)

    # Plot mean ROC
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Classifier', alpha=0.8)
    ax.set_title(f'Mean AUC = {mean_auc:.2f} (±{std_auc:.2f})', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.savefig(f'{scanner}_{task}_AUC.png', dpi=300)
    plt.close()

    return {
        'scanner': scanner,
        'task': task,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_aucs': auc_scores
    }


# 3. Run evaluation
results = []
tasks = ['MCI_vs_NC', 'MCI_vs_AD']
scanners = ['1.5T', '3T']

for scanner in scanners:
    for task in tasks:
        res = evaluate_model(scanner, task, random_state=80)
        results.append(res)
        print(f"{scanner} - {task}: AUC = {res['mean_auc']:.3f} ± {res['std_auc']:.3f}")


# 4. Visualization of results
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=pd.DataFrame(results),
    x='task',
    y='mean_auc',
    hue='scanner',
    palette={'1.5T': 'blue', '3T': 'orange'},
    width=0.6
)
plt.ylabel('AUC Score')
plt.xlabel('Classification Task')
plt.title('Enhanced Model Performance Comparison\n(Feature Selection inside CV + Class Balancing)', fontsize=14, fontweight='bold')
plt.ylim(0.4, 1.05)
plt.grid(True, alpha=0.3)
plt.savefig('scanner_performance_comparison.png', bbox_inches='tight', dpi=300)
plt.show()
