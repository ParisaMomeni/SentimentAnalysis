import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def Regional_analysis(df):
    #fine but we might want to make the output directory a parameter of the function for more flexibility.
    RegionalRoot = f"output/regional/regional"
    state_to_region = {
        'Alabama': 'Southeast',
        'Alaska': 'West',
        'Arizona': 'Southwest',
        'Arkansas': 'Southeast',
        'California': 'West',
        'Colorado': 'West',
        'Connecticut': 'Northeast',
        'Delaware': 'Northeast',
        'Florida': 'Southeast',
        'Georgia': 'Southeast',
        'Hawaii': 'West',
        'Idaho': 'West',
        'Illinois': 'Midwest',
        'Indiana': 'Midwest',
        'Iowa': 'Midwest',
        'Kansas': 'Midwest',
        'Kentucky': 'Southeast',
        'Louisiana': 'Southeast',
        'Maine': 'Northeast',
        'Maryland': 'Northeast',
        'Massachusetts': 'Northeast',
        'Michigan': 'Midwest',
        'Minnesota': 'Midwest',
        'Mississippi': 'Southeast',
        'Missouri': 'Midwest',
        'Montana': 'West',
        'Nebraska': 'Midwest',
        'Nevada': 'West',
        'New Hampshire': 'Northeast',
        'New Jersey': 'Northeast',
        'New Mexico': 'Southwest',
        'New York': 'Northeast',
        'North Carolina': 'Southeast',
        'North Dakota': 'Midwest',
        'Ohio': 'Midwest',
        'Oklahoma': 'Southwest',
        'Oregon': 'West',
        'Pennsylvania': 'Northeast',
        'Rhode Island': 'Northeast',
        'South Carolina': 'Southeast',
        'South Dakota': 'Midwest',
        'Tennessee': 'Southeast',
        'Texas': 'Southwest',
        'Utah': 'West',
        'Vermont': 'Northeast',
        'Virginia': 'Southeast',
        'Washington': 'West',
        'West Virginia': 'Southeast',
        'Wisconsin': 'Midwest',
        'Wyoming': 'West',
        'District of Columbia': 'Northeast',
    }
    #We have a col name Region
    #I can add this part of following code to preprocess_data.py
    df['state_to_region'] = df['Region'].map(state_to_region)
    # Group by 'Region'; calculate mean sentiment score for each region
    #region_group = df.groupby('state_to_region')['sentiment_label'].mean().reset_index()
    region_group = df.groupby('state_to_region')['sentiment_label'].agg([ 'mean', 'std', 'count'])
    print(region_group.head()) 
    region_group = region_group.reset_index()  # This brings 'Gender' back as a column
    print(region_group.head()) 
    region_group.fillna(0, inplace=True)
    region_group.to_csv(f"{RegionalRoot}_mean.csv", index=False, header=True)

    
    # ANOVA: Test if there are significant differences between the regions
    region_sentiment_groups = [df[df['state_to_region'] == region]['sentiment_label'] for region in df['state_to_region'].unique()]
    f_statistic, p_value = stats.f_oneway(*region_sentiment_groups)

    anova_results = pd.DataFrame([{
        'Comparison': 'state_to_region',
        'F-statistic': f_statistic,
        'p-value': p_value
    }])

    # Save ANOVA results to CSV file
    anova_results.fillna(0, inplace=True)
    anova_results.to_csv(f"{RegionalRoot}_anova_results.csv", index=False, header=True)

    # Tukey's HSD test (if significant ANOVA result)
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=df['sentiment_label'],
                                  groups=df['state_to_region'],
                                  alpha=0.05)
        tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_summary.fillna(0, inplace=True)
        tukey_summary.to_csv(f"{RegionalRoot}_tukey_results.csv", index=False, header=True)

    # Perform pairwise t-tests between all regions
    ttest_results = []
    unique_regions = df['state_to_region'].unique()
    
    for i in range(len(unique_regions)):
        for j in range(i + 1, len(unique_regions)):  # Avoid redundant comparisons
            region1 = unique_regions[i]
            region2 = unique_regions[j]

            region1_sentiments = df[df['state_to_region'] == region1]['sentiment_label']
            region2_sentiments = df[df['state_to_region'] == region2]['sentiment_label']

            t_stat, p_val = stats.ttest_ind(region1_sentiments, region2_sentiments, equal_var=True)

            ttest_results.append({
                'region1': region1,
                'region2':  region2,
                't-statistic': t_stat,
                'p-value': p_val
            })
  
    # Save t-test results to CSV
    ttest_results_df = pd.DataFrame(ttest_results)
    ttest_results_df.fillna(0, inplace=True)
    ttest_results_df.to_csv(f"{RegionalRoot}_ttest_results.csv", index=False, header=True)

    print(f"Results saved to {RegionalRoot}_anova_results.csv, {RegionalRoot}_ttest_results.csv")
    return df
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
