import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import csv
import os

def analyze_gender_differences(df):
    # Define output directories
    output_dir = "output/gender/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = df.dropna(subset=['Gender', 'sentiment_label'])

    # Group by Gender and calculate mean sentiment scores
    #gender_groups = df.groupby('Gender')['sentiment_label']
    #means = gender_groups.mean()
    #stds = gender_groups.std()
    #means.fillna(0, inplace=True)
    #stds.fillna(0, inplace=True)
    #means.to_csv(f"{output_dir}gender_mean_stats.csv")
    #stds.to_csv(f"{output_dir}gender_std_stats.csv")
    # Save descriptive statistics
    gender_stats = df.groupby('Gender')['sentiment_label'].agg([ 'mean', 'std', 'count'])
    print(gender_stats.head()) 
    gender_stats = gender_stats.reset_index()  # This brings 'Gender' back as a column
    print(gender_stats.head()) 
    gender_stats.fillna(0, inplace=True)
    gender_stats.to_csv(f"{output_dir}gender_descriptive_stats.csv", index=False, header=True)

    # Pairwise t-tests
    male_sentiments = df[df['Gender'] == 'male']['sentiment_label']
    female_sentiments = df[df['Gender'] == 'female']['sentiment_label']
    unknown_sentiments = df[df['Gender'] == 'unknown']['sentiment_label']

    t_stat_mf, p_value_mf = ttest_ind(male_sentiments, female_sentiments, equal_var=False)
    t_stat_mu, p_value_mu = ttest_ind(male_sentiments, unknown_sentiments, equal_var=False)
    t_stat_fu, p_value_fu = ttest_ind(female_sentiments, unknown_sentiments, equal_var=False)

    # Write t-test results to CSV
    with open(f"{output_dir}gender_ttest_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['genderGroup1', 'genderGroup2', 't-statistic', 'p-value'])
        writer.writerow(['Men', 'Women', t_stat_mf, p_value_mf])
        writer.writerow(['Men', 'Unknown', t_stat_mu, p_value_mu])
        writer.writerow(['Women', 'Unknown', t_stat_fu, p_value_fu])

    # One-Way ANOVA
    model = ols('sentiment_label ~ C(Gender)', data=df).fit()
    anova_table = anova_lm(model, typ=2)

    # Save ANOVA results to CSV
    anova_table.fillna(0, inplace=True)
    anova_table.to_csv(f"{output_dir}gender_anova_results.csv", index=False, header=True)

    # Post-hoc Test: Tukey's HSD
    tukey = pairwise_tukeyhsd(df['sentiment_label'], df['Gender'])

    # Convert Tukey results to a DataFrame and save to CSV
    tukey_results = pd.DataFrame(
        data=tukey.summary().data[1:],  # Exclude header
        columns=tukey.summary().data[0]  # Use header from the Tukey summary
    )
    tukey_results.fillna(0, inplace=True)
    tukey_results.to_csv(f"{output_dir}gender_tukey_results.csv", index=False, header=True)

    print("Analysis complete. Results saved to the 'output/gender/' directory.")
