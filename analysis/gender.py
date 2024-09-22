import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

def analyze_gender_differences(df, output_folder):
    # Ensure the output folder exists
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # T-test between male and female sentiments
    male = df[df['Gender'] == 'Male']['sentiment_label']
    female = df[df['Gender'] == 'Female']['sentiment_label']
    unknown_sentiments = df[df['Gender'] == 'unknown']['sentiment_label']
    t_stat_mf, p_value_mf = ttest_ind(male_sentiments, female_sentiments, equal_var=False)
    t_stat_mu, p_value_mu = ttest_ind(male_sentiments, unknown_sentiments, equal_var=False)
    t_stat_fu, p_value_fu = ttest_ind(female_sentiments, unknown_sentiments, equal_var=False)
    
 
    # Save t-test results
    with open(f"output/gender/gender_ttest_results.txt", 'w') as f:
        f.write("\nT-test Results:")
        f.write(f"Men vs Women: t-statistic = {t_stat_mf}, p-value = {p_value_mf}\n")
        f.write(f"Men vs Unknown: t-statistic = {t_stat_mu}, p-value = {p_value_mu}\n")
        f.write(f"Women vs Unknown: t-statistic = {t_stat_fu}, p-value = {p_value_fu}\n")

    # ANOVA
    model = ols('sentiment_label ~ C(Gender)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"output/gender/gender_anova_results.csv")
  # Post-hoc Test: Tukey's HSD
    tukey = pairwise_tukeyhsd(df['sentiment_label'], df['Gender'])
    f.write("\nTukey's HSD Post-hoc Test Results:")
    f.write(tukey)
  

    
    # Descriptive statistics
    gender_stats = df.groupby('Gender')['sentiment_label'].agg(['mean', 'std', 'count'])
    gender_stats.to_csv(f"output/gender/gender_descriptive_stats.csv")

    # Visualizations
    
    # 1. Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Gender')
    plt.savefig(f"{output_folder}/gender_sentiment_boxplot.png")
    plt.close()

    # 2. Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Gender (Violin Plot)')
    plt.savefig(f"{output_folder}/gender_sentiment_violinplot.png")
    plt.close()

    # 3. Bar plot of mean sentiments
    plt.figure(figsize=(10, 6))
    gender_stats['mean'].plot(kind='bar', yerr=gender_stats['std'], capsize=5)
    plt.title('Mean Sentiment by Gender')
    plt.ylabel('Mean Sentiment')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/gender_mean_sentiment_barplot.png")
    plt.close()

    # 4. Histogram of sentiments for each gender
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='sentiment_label', hue='Gender', element='step', stat='density', common_norm=False)
    plt.title('Distribution of Sentiments by Gender')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.savefig(f"{output_folder}/gender_sentiment_histogram.png")
    plt.close()

    print(f"Gender analysis complete. Results saved in {output_folder}")

if __name__ == "__main__":
    # This allows you to run this analysis independently if needed
    import sys
    if len(sys.argv) != 3:
        print("Usage: python gender.py <grouped_data_file> <output_folder>")
        sys.exit(1)

    grouped_data_file = sys.argv[1]
    output_folder = sys.argv[2]

    print(f"Loading grouped data from {grouped_data_file}...")
    grouped_by_author_df = pd.read_pickle(grouped_data_file)

    analyze_gender_differences(grouped_by_author_df, output_folder)