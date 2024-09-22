import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

def Verified_users(df, output_folder):
    # Ensure the output folder exists
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Separate verified and non-verified users
    verified = df[df['Verified'] == True]['sentiment_label']
    not_verified = df[df['Verified'] == False]['sentiment_label']

    # Perform t-test
    t_stat, p_val = stats.ttest_ind(verified, not_verified)

    # Save t-test results
    with open(f"{output_folder}/verification_ttest_results.txt", 'w') as f:
        f.write(f"Verified vs Non-Verified Users Analysis:\n")
        f.write(f"T-statistic: {t_stat}\n")
        f.write(f"P-value: {p_val}\n")

    # Perform one-way ANOVA
    model = ols('sentiment_label ~ C(Verified)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"{output_folder}/verification_anova_results.csv")

    # Calculate descriptive statistics
    verification_stats = df.groupby('Verified')['sentiment_label'].agg(['mean', 'std', 'count'])
    verification_stats.to_csv(f"{output_folder}/verification_descriptive_stats.csv")

    # Visualizations

    # 1. Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Verified', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Verification Status')
    plt.savefig(f"{output_folder}/verification_sentiment_boxplot.png")
    plt.close()

    # 2. Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Verified', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Verification Status (Violin Plot)')
    plt.savefig(f"{output_folder}/verification_sentiment_violinplot.png")
    plt.close()

    # 3. Bar plot of mean sentiments
    plt.figure(figsize=(10, 6))
    verification_stats['mean'].plot(kind='bar', yerr=verification_stats['std'], capsize=5)
    plt.title('Mean Sentiment by Verification Status')
    plt.ylabel('Mean Sentiment')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/verification_mean_sentiment_barplot.png")
    plt.close()

    # 4. Histogram of sentiments for verified and non-verified users
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='sentiment_label', hue='Verified', element='step', stat='density', common_norm=False)
    plt.title('Distribution of Sentiments by Verification Status')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.savefig(f"{output_folder}/verification_sentiment_histogram.png")
    plt.close()

    # 5. Pie chart of verified vs non-verified users
    plt.figure(figsize=(10, 6))
    df['Verified'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Proportion of Verified vs Non-Verified Users')
    plt.ylabel('')
    plt.savefig(f"{output_folder}/verification_proportion_piechart.png")
    plt.close()

    # 6. Scatter plot of sentiment vs. tweet count, colored by verification status
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='tweet_count', y='sentiment_label', hue='Verified', alpha=0.6)
    plt.title('Sentiment vs. Tweet Count by Verification Status')
    plt.xlabel('Tweet Count')
    plt.ylabel('Sentiment Score')
    plt.savefig(f"{output_folder}/verification_sentiment_vs_tweetcount_scatter.png")
    plt.close()

    print(f"Verified users analysis complete. Results saved in {output_folder}")

if __name__ == "__main__":
    # This allows you to run this analysis independently if needed
    import sys
    if len(sys.argv) != 3:
        print("Usage: python verified_users.py <grouped_data_file> <output_folder>")
        sys.exit(1)

    grouped_data_file = sys.argv[1]
    output_folder = sys.argv[2]

    print(f"Loading grouped data from {grouped_data_file}...")
    grouped_by_author_df = pd.read_pickle(grouped_data_file)

    Verified_users(grouped_by_author_df, output_folder)