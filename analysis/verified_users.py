import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os
def Verified_users(df):
    print(df.head())
    print(df.shape)
    pd.set_option('display.max_rows', 20)  # Adjust this number to show more or fewer rows


    grouped_by_author_df = df.groupby('Author').agg({
    'Twitter Verified': lambda x: np.any(x),  # True if any tweet is verified #If an author has any tweet where 'Twitter Verified' is True, the result for that author will be True.
    'sentiment_label': 'mean'
    }).reset_index()
    print(grouped_by_author_df.head())
    output_root = "output/verified_users"  
    print("Available columns:")
    for col in grouped_by_author_df:
        print(col)
    # Separate verified and non-verified users
    verified = grouped_by_author_df[grouped_by_author_df['Twitter Verified'] == True]['sentiment_label']
    not_verified = grouped_by_author_df[grouped_by_author_df['Twitter Verified'] == False]['sentiment_label']
#mean of verified and not verified
    verified_mean = verified.mean()
    not_verified_mean = not_verified.mean()
    verified_median = verified.median()
    not_verified_median = not_verified.median()
    verified_std = verified.std()
    not_verified_std = not_verified.std()
    verified_count = verified.count()
    not_verified_count = not_verified.count()
    
    all_stats = pd.DataFrame({
        'User Type': ['Verified', 'Not Verified'],
        'Mean': [verified_mean, not_verified_mean],
        'Median': [verified_median, not_verified_median],
        'Std Dev': [verified_std, not_verified_std],
        'Count': [verified_count, not_verified_count]
    })

# Save combined statistics to CSV
    all_stats.to_csv(f"{output_root}/verified_user_stats.csv", index=False, header=True)
    # Perform t-test
   # t_stat, p_val = stats.ttest_ind(verified, not_verified)

    # Save t-test results
   # with open(f"{output_root}/verification_ttest_results.txt", 'w') as f:
        #f.write(f"Verified vs Non-Verified Users Analysis:\n")
        #f.write(f"T-statistic: {t_stat}\n")
        #f.write(f"P-value: {p_val}\n")

    # Perform one-way ANOVA
    #model = ols('sentiment_label ~ C(Twitter_Verified)', data=df).fit()
    #anova_table = sm.stats.anova_lm(model, typ=2)
    #anova_table.to_csv(f"{output_root}/verification_anova_results.csv")

    # Calculate descriptive statistics
    #verification_stats = df.groupby('Twitter_Verified')['sentiment_label'].agg(['mean', 'std', 'count'])
    #verification_stats.to_csv(f"{output_root}/verification_descriptive_stats.csv")

    print(f"Verified users analysis complete. Results saved in {output_root}")

