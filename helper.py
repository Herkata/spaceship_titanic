import nbformat
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

def test_proportion_difference_two_tail(data, group_column, target_column, group1, group2, confidence=0.95):
    """
    Conducts a statistical test for the difference in proportions of the target variable between two groups.
    
    Parameters:
    - data: DataFrame containing the data.
    - group_column: Column name that defines the groups (e.g., spending category).
    - target_column: Column name of the binary target variable.
    - group1: The first group to compare.
    - group2: The second group to compare.
    - confidence: Confidence level for the confidence interval (default: 0.95).
    
    Returns:
    - Dictionary with proportion difference, p-value, and confidence interval.
    """
    # Count occurrences
    n1 = data[data[group_column] == group1].shape[0]  # Sample size for group1
    n2 = data[data[group_column] == group2].shape[0]  # Sample size for group2
    
    if n1 == 0 or n2 == 0:
        raise ValueError("One of the groups has zero samples.")
    
    p1 = data.loc[data[group_column] == group1, target_column].mean()  # Proportion of positive cases in group1
    p2 = data.loc[data[group_column] == group2, target_column].mean()  # Proportion of positive cases in group2
    
    # Proportion difference
    prop_diff = p1 - p2
    
    # Standard error for difference in proportions
    pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)  # Pooled proportion
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    
    # Z-score and p-value
    z_score = prop_diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    
    # Confidence interval
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin_of_error = z_crit * se
    conf_interval = (prop_diff - margin_of_error, prop_diff + margin_of_error)
    
    print(f"Proportion Difference: {round(prop_diff, 4)}")
    print(f"Z-Score: {z_score:.4f}")
    print(f"P-Value: {p_value:.4e}")
    print(f"{int(confidence * 100)}% Confidence Interval: ({round(conf_interval[0], 4)}, {round(conf_interval[1], 4)})")
    
    if p_value < (1 - confidence):
        print("Reject the null hypothesis: There is a significant difference in proportions.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in proportions.")


def test_proportion_difference_one_tail(df, group_col, target_col, baseline_group, alternative_groups, alternative="greater", alpha=0.05):
    """
    Conducts a Z-test for difference in proportions and calculates the confidence interval.

    Parameters:
    - df: DataFrame containing the data
    - group_col: Column representing categorical groups (e.g., 'HomePlanet')
    - target_col: Binary target column (e.g., 'Transported', where 1 = transported, 0 = not)
    - baseline_group: The reference group (e.g., 'Earth')
    - alternative_groups: A single group (str) or multiple groups (list) for comparison
    - alternative: Type of test ('greater', 'less', or 'two-sided')
    - alpha: Significance level (default = 0.05 for 95% confidence interval)

    Returns:
    - p-value, confidence interval, and proportion difference
    """
    # Ensure alternative_groups is a list
    if isinstance(alternative_groups, str):
        alternative_groups = [alternative_groups]

    # Baseline group stats
    n1 = df[df[group_col] == baseline_group][target_col].count()   # Total count in baseline group
    x1 = df[df[group_col] == baseline_group][target_col].sum()     # Transported count in baseline group
    p1 = x1 / n1  # Proportion transported

    # Alternative group stats (aggregate if multiple groups)
    alt_df = df[df[group_col].isin(alternative_groups)]
    n2 = alt_df[target_col].count()  # Total count in alternative groups
    x2 = alt_df[target_col].sum()    # Transported count in alternative groups
    p2 = x2 / n2  # Proportion transported

    prop_diff = p1 - p2  # Difference in proportions

    # Pooled proportion for standard error calculation
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Z-score
    z_score = prop_diff / se

    # Determine p-value based on test type
    if alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_score)  # Right-tailed test (p1 > p2)
    elif alternative == "less":
        p_value = stats.norm.cdf(z_score)  # Left-tailed test (p1 < p2)
    else:
        raise ValueError("Invalid alternative hypothesis. Choose 'greater', 'less', or 'two-sided'.")

    # Confidence interval (always two-tailed)
    z_critical = stats.norm.ppf(1 - alpha/2)
    margin_of_error = z_critical * se
    conf_interval = (prop_diff - margin_of_error, prop_diff + margin_of_error)

    # Print results
    print(f"Proportion ({baseline_group}): {p1:.4f}, Proportion (Alternative Groups {alternative_groups}): {p2:.4f}")
    print(f"Proportion Difference: {prop_diff:.4f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"P-Value: {p_value:.4e}")
    print(f"95% Confidence Interval: {conf_interval}")

    if p_value < alpha:
        print(f"Reject the null hypothesis: The proportion is {alternative}.")
    else:
        print(f"Fail to reject the null hypothesis: The proportion is not {alternative}.")


def clean_notebook(notebook_path: str) -> str:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    imports = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"].split("\n")
            new_lines = []
            for line in lines:
                if re.match(r"^\s*(import|from)\s+\S+", line):
                    imports.append(line)
                elif not re.match(r"^\s*#", line):
                    new_lines.append(line.split("#")[0].rstrip())  
            cell["source"] = "\n".join(new_lines).strip()

    while len(nb["cells"]) < 4:
        nb["cells"].append(nbformat.v4.new_code_cell(""))

    nb["cells"][3]["source"] = "\n".join(sorted(set(imports))) 

    cleaned_notebook_path = "cleaned_" + notebook_path
    with open(cleaned_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    return cleaned_notebook_path


def describe_dataset(df: pd.DataFrame) -> None:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("="*40)
    print("🔍 Dataset Overview")
    print("="*40)

    print("\n📌 Data Types:")
    print(df.dtypes.to_string())

    print("\n📌 Number of Rows and Columns:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n📌 Duplicate Rows:")
    print(df.duplicated().sum())

    print("\n📌 Missing Values:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
    print(missing_df[missing_df['Missing Values'] > 0].to_string())

    print("\n📌 Summary Statistics (Numerical Features):")
    print(df.describe().to_string())

    print("\n📌 Summary Statistics (Categorical Features):")
    print(df.describe(include=['O']).to_string())

    print("="*40)

def plot_feature_distribution(data, feature, y_scale='normal'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    transformed_feature = np.log1p(data[feature]) if y_scale == 'log' else data[feature]

    sns.boxplot(y=transformed_feature, ax=axes[0])
    axes[0].set_title(f'Boxplot of {feature} ({y_scale} scale)')
    axes[0].set_ylabel(f'{feature} (log scale)' if y_scale == 'log' else feature)

    sns.histplot(y=transformed_feature, kde=True, ax=axes[1])
    axes[1].set_title(f'Histogram of {feature} ({y_scale} scale)')
    axes[1].set_ylabel(f'{feature} (log scale)' if y_scale == 'log' else feature)

    plt.tight_layout()
    plt.show()

def plot_feature_distribution_split(data, feature, y_scale='normal'):
    plot_df = data.copy()  
    zero_count = (plot_df[feature] == 0).sum()
    non_zero_data = plot_df[plot_df[feature] > 0]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    sns.barplot(x=['Zero', 'Non-Zero'], y=[zero_count, len(non_zero_data)], ax=axes[0])
    axes[0].set_title(f'Zero vs. Non-Zero Counts ({feature})')
    axes[0].set_ylabel('Count')

    if y_scale == 'log':
        non_zero_data.loc[:, feature] = np.log1p(non_zero_data[feature]) 

    sns.boxplot(data=non_zero_data, y=feature, ax=axes[1])
    axes[1].set_title(f'Boxplot of Non-Zero {feature}')
    axes[1].set_ylabel(f'{"Log(1 + " + feature + ")" if y_scale == "log" else feature}')

    sns.histplot(non_zero_data, y=feature, kde=True, ax=axes[2])
    axes[2].set_title(f'Histogram of Non-Zero {feature}')
    axes[2].set_ylabel(f'{"Log(1 + " + feature + ")" if y_scale == "log" else feature}')

    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(data, cat_features, target):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    axes = axes.flatten()

    for i, feature in enumerate(cat_features + [target]):
        data[feature].value_counts().plot.pie(
            autopct=lambda p: f'{p:.1f}%\n({int(p * len(data) / 100)})', 
            startangle=90, 
            counterclock=False, 
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_ylabel('')

    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

def split_cabin(df):
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop(columns=['Cabin'])

def jaccard_similarity(row1, row2, categorical_cols):
    """Compute Jaccard similarity between two categorical records."""
    matches = sum(row1[col] == row2[col] for col in categorical_cols)
    return matches / len(categorical_cols)

def compute_pairwise_similarity(group, numeric_cols, categorical_cols, num_weight=0.7, cat_weight=0.3):
    """Compute similarity for all pairs within a group (only records with the same Name)."""
    if len(group) < 2:
        return None  # Skip if only one record with that Name

    similarities = []
    
    # Ensure numerical columns have no NaNs (fill with median of the group)
    group.loc[:, numeric_cols] = group[numeric_cols].fillna(group[numeric_cols].median())

    # Generate all possible pairs within the group
    for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
        # Compute cosine similarity for numerical features
        num_vector1 = row1[numeric_cols].values.reshape(1, -1)
        num_vector2 = row2[numeric_cols].values.reshape(1, -1)
        num_sim = cosine_similarity(num_vector1, num_vector2)[0][0]  # Extract single similarity value

        # Compute Jaccard similarity for categorical features
        cat_sim = jaccard_similarity(row1, row2, categorical_cols)

        # Compute final similarity score
        final_sim = (num_sim * num_weight) + (cat_sim * cat_weight)

        similarities.append({
            'Name': row1['Name'],  # Keep track of which Name these belong to
            'Index1': idx1, 'Index2': idx2,
            'Numerical_Similarity': num_sim,
            'Categorical_Similarity': cat_sim,
            'Final_Similarity': final_sim
        })

    return pd.DataFrame(similarities)

def process_groups(df):
    df = df.copy()
    df['GroupSize'] = df.groupby(df['PassengerId'].str.split('_').str[0])['PassengerId'].transform('count')
    df['InGroup'] = df['GroupSize'] > 1
    return df.drop(columns=['PassengerId'])

def drop_name(df):
    """Drops the 'Name' column."""
    return df.drop(columns=['Name'], errors='ignore')

def split_cabin(df):
    """Splits the 'Cabin' column into 'Deck', 'Num', and 'Side'."""
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop(columns=['Cabin'])

def impute_cryosleep(df):
    """Infers 'CryoSleep' based on billing columns."""
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['CryoSleep'] = df['CryoSleep'].fillna(df[billing_cols].sum(axis=1) == 0)
    return df

def impute_billing(df):
    """Imputes missing billing values based on 'CryoSleep' status."""
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    df.loc[df['CryoSleep'], billing_cols] = 0  

    df[billing_cols] = df[billing_cols].fillna(df[billing_cols].median())

    return df

def sum_billing(df):
    """Creates a 'TotalBill' column as the sum of all billing-related columns."""
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalBill'] = df[billing_cols].sum(axis=1)
    return df

def convert_bool(df):
    """Converts 'CryoSleep' and 'VIP' columns to boolean dtype."""
    df[['CryoSleep', 'VIP']] = df[['CryoSleep', 'VIP']].astype('boolean')
    return df

def test_proportion_difference_zero_vs_nonzero(df, column, target, alternative='greater', alpha=0.05):
    """
    Conducts a proportion difference test comparing passengers who spent **zero** vs. those who spent **something** 
    on a specific category (e.g., 'Spa').

    Parameters:
    - df: DataFrame containing the data.
    - column: Column used for splitting (e.g., 'Spa').
    - target: Boolean target column (e.g., 'Transported' with 1 = transported).
    - alternative: 
        'greater' -> Nonzero spenders have a higher proportion of transport than zero spenders.
        'less' -> Nonzero spenders have a lower proportion of transport than zero spenders.
        'two-sided' -> There is some difference.
    - alpha: Significance level (default 0.05).

    Returns:
    - Z-score, p-value, and confidence interval for the proportion difference.
    """
    # Define groups
    zero_spenders = df[df[column] == 0]
    nonzero_spenders = df[df[column] > 0]

    # Calculate proportions
    p1 = zero_spenders[target].mean()  # Proportion transported (zero spenders)
    p2 = nonzero_spenders[target].mean()  # Proportion transported (nonzero spenders)

    # Sample sizes
    n1, n2 = len(zero_spenders), len(nonzero_spenders)

    # Standard error
    p_pool = (zero_spenders[target].sum() + nonzero_spenders[target].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    # Z-score
    z_score = (p1 - p2) / se

    # P-value
    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_score)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_score)
    else:  # 'two-sided'
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Confidence interval for difference
    margin = stats.norm.ppf(1 - alpha/2) * se
    conf_int = ((p1 - p2) - margin, (p1 - p2) + margin)

    print(f"Proportion Zero Spenders: {p1:.4f} out of {n1}")
    print(f"Proportion Nonzero Spenders: {p2:.4f} out of {n2}")
    print(f"Proportion Difference: {p1 - p2:.4f}")
    print(f"95% Confidence Interval: ({conf_int[0]:.4f}, {conf_int[1]:.4f})")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.4e}")
    