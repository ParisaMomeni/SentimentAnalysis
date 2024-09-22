import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

def analyze_interest_users(df, output_folder):
    # Ensure the output folder exists
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Calculate statistics for each interest
    interest_stats = df.groupby('Interest')['sentiment_label'].agg(['mean', 'std', 'count'])
    interest_stats = interest_stats.sort_values('mean', ascending=False)

    # Save interest statistics
    interest_stats.to_csv(f"{output_folder}/interest_statistics.csv")

    # Perform one-way ANOVA
    model = ols('sentiment_label ~ C(Interest)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"{output_folder}/interest_anova_results.csv")

    # Perform Tukey's HSD test for pairwise comparisons
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey_results = pairwise_tukeyhsd(df['sentiment_label'], df['Interest'])
    with open(f"{output_folder}/tukey_hsd_results.txt", 'w') as f:
        f.write(str(tukey_results))

    # Visualizations

    # 1. Bar plot of mean sentiments by interest
    plt.figure(figsize=(12, 6))
    interest_stats['mean'].plot(kind='bar', yerr=interest_stats['std'], capsize=5)
    plt.title('Mean Sentiment by Interest')
    plt.xlabel('Interest')
    plt.ylabel('Mean Sentiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/interest_mean_sentiment_barplot.png")
    plt.close()

    # 2. Box plot of sentiments by interest
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Interest', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Interest')
    plt.xlabel('Interest')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/interest_sentiment_boxplot.png")
    plt.close()

    # 3. Heatmap of sentiment correlation between interests
    pivot_df = df.pivot_table(values='sentiment_label', index='Author', columns='Interest', aggfunc='first')
    correlation_matrix = pivot_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation of Sentiments Between Interests')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/interest_correlation_heatmap.png")
    plt.close()

    # 4. Violin plot of sentiments by interest
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Interest', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Interest (Violin Plot)')
    plt.xlabel('Interest')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/interest_sentiment_violinplot.png")
    plt.close()

    # 5. Scatter plot of sentiment vs. number of users for each interest
    plt.figure(figsize=(12, 8))
    plt.scatter(interest_stats['count'], interest_stats['mean'], alpha=0.6)
    for idx, row in interest_stats.iterrows():
        plt.annotate(idx, (row['count'], row['mean']))
    plt.title('Mean Sentiment vs. Number of Users by Interest')
    plt.xlabel('Number of Users')
    plt.ylabel('Mean Sentiment')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/interest_sentiment_vs_users_scatter.png")
    plt.close()

    print(f"Interest analysis complete. Results saved in {output_folder}")

if __name__ == "__main__":
    # This allows you to run this analysis independently if needed
    import sys
    if len(sys.argv) != 3:
        print("Usage: python interests.py <grouped_data_file> <output_folder>")
        sys.exit(1)

    grouped_data_file = sys.argv[1]
    output_folder = sys.argv[2]

    print(f"Loading grouped data from {grouped_data_file}...")
    grouped_by_author_df = pd.read_pickle(grouped_data_file)

    analyze_interest_users(grouped_by_author_df, output_folder)