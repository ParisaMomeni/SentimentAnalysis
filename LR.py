#I plan to add multi threading to this code to speed up the process. no 
# I want to extract most diffrent groups and visualize the results. Since visualization of 52 * 52 groups is not possible.---> Compare E W #population size

# 1. save results on output file that be readable by df = pd.read_csv('data.csv') 2. ANALYZE AND VISUALIZE THE RESULTS
# MAKE roberta sentiment seperate file to save time
# variius sentiment per user?
# print # of rows in each run at top of the output file
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import torch
from scipy.special import softmax
import sys
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.impute import SimpleImputer
import re
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats

def visualize_sentiment_analysis(df, sentiment, independent_var, ttest_p_values, anova_model=None, ):
    """
    Generalized function to visualize sentiment analysis results across different independent variables.

    Args:
    df (pd.DataFrame): Dataframe containing sentiment and independent variables.
    sentiment (str): Column name for sentiment label.
    independent_var (str): Column name for the independent variable (e.g., 'Gender', 'Location', 'Interest').
    ttest_p_values (tuple): Tuple containing p-values for t-test comparisons.
    anova_model (statsmodels): Fitted model object from ANOVA analysis (optional).
    """
    # Boxplot
    plot_sentiment_boxplots(df, sentiment, independent_var, save_path=None)
    
    # Barplot
    plot_mean_sentiment_bars(df, sentiment, independent_var, save_path=None)
    
    # T-test Results
    plot_ttest_results(ttest_p_values,sentiment, independent_var, save_path=None)
    
    # ANOVA residuals if available
    if anova_model:
        plot_anova_residuals(anova_model,sentiment, independent_var, save_path=None)
    
    # Tukey HSD
    plot_tukey_hsd(df, sentiment, independent_var, save_path=None)

def plot_sentiment_boxplots(df, sentiment, independent_var, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=independent_var, y=sentiment, data=df)
    plt.title(f'Boxplot of {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.ylabel(f'{sentiment.replace("_score", "").capitalize()} Sentiment Score')
    plt.xlabel(independent_var)
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_mean_sentiment_bars(df, sentiment, independent_var, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=independent_var, y=sentiment, data=df, errorbar=None)
    plt.title(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.ylabel(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment Score')
    plt.xlabel(independent_var)
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()


def plot_ttest_results(p_values,sentiment, independent_var, save_path=None):
    ttest_results = {
        "Comparison": ["Comparison 1", "Comparison 2", "Comparison 3"],  # Update as necessary
        "p-value": p_values
    }
    ttest_df = pd.DataFrame(ttest_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Comparison", y="p-value", data=ttest_df)
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
    plt.title('Pairwise T-test Results')
    plt.ylabel('p-value')
    plt.legend()
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_anova_residuals(model, sentiment, independent_var, save_path=None):
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()

def plot_tukey_hsd(df, sentiment, independent_var, save_path=None):
    tukey = pairwise_tukeyhsd(df[sentiment], df[independent_var])
    tukey.plot_simultaneous()
    plt.title(f'Tukey HSD Test for {sentiment.replace("_score", "").capitalize()} Sentiment by {independent_var}')
    plt.xlabel('Mean Difference')
    if save_path:
        plt.savefig(f'{save_path}/boxplot_{sentiment}_{independent_var}.png')
    plt.show()


##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------   

def Regional_analysis(df, RegionalTtestOutput, RegionalAnovaOutput):
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
        'Puerto Rico': 'Territory'
    }
    df['Region'] = df['State'].map(state_to_region)
    # Group by 'Region' and calculate mean sentiment score for each region
    region_group = df.groupby('Region')['sentiment_label'].mean().reset_index()

    # ANOVA: Test if there are significant differences between the regions
    region_sentiment_groups = [df[df['Region'] == region]['sentiment_label'] for region in df['Region'].unique()]
    f_statistic, p_value = stats.f_oneway(*region_sentiment_groups)

    anova_results = pd.DataFrame([{
        'Comparison': 'Regions',
        'F-statistic': f_statistic,
        'p-value': p_value
    }])

    # Save ANOVA results to CSV file
    anova_results.to_csv(f"{RegionalAnovaOutput}_anova_results.csv", index=True)

    # Tukey's HSD test (if significant ANOVA result)
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=df['sentiment_label'],
                                  groups=df['Region'],
                                  alpha=0.05)
        tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_summary.to_csv(f"{RegionalAnovaOutput}_tukey_results.csv", index=True)

    # Perform pairwise t-tests between all regions
    ttest_results = []
    unique_regions = df['Region'].unique()
    
    for i in range(len(unique_regions)):
        for j in range(i + 1, len(unique_regions)):  # Avoid redundant comparisons
            region1 = unique_regions[i]
            region2 = unique_regions[j]

            region1_sentiments = df[df['Region'] == region1]['sentiment_label']
            region2_sentiments = df[df['Region'] == region2]['sentiment_label']

            t_stat, p_val = stats.ttest_ind(region1_sentiments, region2_sentiments, equal_var=False)

            ttest_results.append({
                'Comparison': f"{region1} vs {region2}",
                't-statistic': t_stat,
                'p-value': p_val
            })

    # Save t-test results to CSV
    ttest_results_df = pd.DataFrame(ttest_results)
    ttest_results_df.to_csv(f"{RegionalTtestOutput}_ttest_results.csv", index=True)

    print(f"Results saved to {RegionalTtestOutput}_anova_results.csv, {RegionalTtestOutput}_ttest_results.csv")


##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
def Verifiedـusers(df, AccountTypeTtestOutput, AccountTypeAnovaOutput):
    #group by verified users. Col= 'Twitter Verified'
    #true or false
    #extract sentiment scores for each group
    #Group by 'Twitter Verified' status
    verified_df = df[df['Twitter Verified'] == True]
    non_verified_df = df[df['Twitter Verified'] == False]
    
    verified_sentiments = verified_df['sentiment_label']
    non_verified_sentiments = non_verified_df['sentiment_label']
    #??
    # t-test between verified and non-verified users
    t_stat, p_val = ttest_ind(verified_sentiments, non_verified_sentiments, equal_var=False)
    
    # Save t-test result to CSV file
    ttest_results = pd.DataFrame([{
        'Comparison': 'Verified vs Non-Verified',
        't-statistic': t_stat,
        'p-value': p_val
    }])
    ttest_results.to_csv(AccountTypeTtestOutput, mode='w', header=True, index=True)

    # Perform ANOVA: For a single factor (verified status), it's a simple comparison
    f_statistic, p_value = stats.f_oneway(verified_sentiments, non_verified_sentiments)
    
    # Save ANOVA result to CSV file
    anova_results = pd.DataFrame([{
        'Comparison': 'Verified vs Non-Verified',
        'F-statistic': f_statistic,
        'p-value': p_value
    }])
    
    # Check if the file exists, if not create it with headers, otherwise append data
   
    anova_results.to_csv(AccountTypeAnovaOutput, mode='w', header=True, index=True)

    # Print results for verification
    anova_results.to_csv(AccountTypeAnovaOutput, mode='a', header=not pd.io.common.file_exists(AccountTypeAnovaOutput), index=False)
    ttest_results.to_csv(AccountTypeTtestOutput, mode='a', header=not pd.io.common.file_exists(AccountTypeAnovaOutput), index=False)

    print(f"T-test results saved to {AccountTypeAnovaOutput}")
    print(f"ANOVA results saved to {AccountTypeAnovaOutput}")

# Example usage
# df = pd.read_csv('your_data_file.csv')  # Replace with your actual file path
# Verified_users(df, 'AccoutnTypeTtestOutput.csv', 'AccoutnTypeAnovaOutput.csv')
# ##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------   

# Sentiment Analysis by Interest Groups: This function groups users by their interests and calculates the average sentiment for each group. It performs pairwise t-tests between different interest groups to check for significant differences in sentiment scores.
# The function groups users by their interests and calculates the average sentiment for each group.
# It performs pairwise t-tests between different interest groups to check. Is that mean of each interest group? 
def analyze_interest_users(df, fileOutput, fileOutputName):
    #Group by users by their interest categories: What if a user has multiple interests? consider the first one?  or might average across all interests 
    #Perform hypothesis testing: Are there significant differences in sentiment scores between different interest categories?
    # I want to group by interest and calculate the average sentiment score for each group.
    # for each user I have a sentiment score.
    #SELECT Interest, AVG(sentiment_label) AS avg_sentiment
        #FROM df
        #GROUP BY Interest;
    pd.set_option('display.max_rows', None)  # Set this to None to display all rows
    df['Interest'] = df['Interest'].fillna('Unknown')
    avg_sentiment_per_interest = df.groupby('Interest')['sentiment_label'].mean()
    #user_grouped_by_interest = df.groupby('Interest')
    # Step 2: Calculate the average sentiment scores per interest group
    #avg_sentiment_per_group = user_grouped_by_interest.mean()
    print("\nAverage Sentiment Scores per Interest Group:")
    for interest, avg_sentiment in avg_sentiment_per_interest.items():
        #print(f"Interest: {interest}, Average Sentiment Score: {avg_sentiment:.4f}")
        if fileOutput == 1:
            with open(fileOutputName, "w") as outputFile:
                outputFile.write(f"Interest: {interest}, Average Sentiment Score: {avg_sentiment:.4f}\n")
                outputFile.write(f"-------------------------------------------------------------------------------------------\n")

    # Step 3: Extract interest groups
    #interest_groups = list(avg_sentiment_per_interest.groups.keys())
    interest_groups = list(avg_sentiment_per_interest.index)

    results = {}
    
    # Step 4: Perform hypothesis testing for sentiment score differences between interest groups
    for i, group1 in enumerate(interest_groups):
        for j, group2 in enumerate(interest_groups):
            if i < j:
                # Extract sentiment scores for each interest group
                # group2_scores = df[df[Interest] == group2][sentiment_label]
                group1_scores = df[df['Interest'] == group1]['sentiment_label']
                group2_scores = df[df['Interest'] == group2]['sentiment_label']
                
                # Perform t-test
                #equal_var=False: two groups are assumed to have unequal variances (Welch's t-test).
                #True, equal variances (the standard t-test).
                #t-test: used to determine if there is a significant difference between the means of two groups
                t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False)
                
                # Store the results
                results[f"{group1} vs {group2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value
                }
                
                # Print the results
                
                #print(f"Hypothesis Test: {group1} vs {group2} -> t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                if fileOutput == 1:
                    with open(fileOutputName, "a") as outputFile:
                        outputFile.write(f"Hypothesis Test: {group1} vs {group2} -> t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}\n")           
                        outputFile.write(f"-------------------------------------------------------------------------------------------\n")


    # Prepare data for ANOVA
    # Group sentiment scores by interest 
    grouped_data = [df[df['Interest'] == group]['sentiment_label'].values for group in interest_groups]

    # Perform ANOVA
    # f_oneway: used to determine if there are any statistically significant differences between the means of two or more independent (unrelated) groups.
    f_statistic, p_value = stats.f_oneway(*grouped_data)
    
    # Print ANOVA results
    #print(f"\nANOVA Results -> F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
    if fileOutput == 1:
        with open(fileOutputName, "a") as outputFile:
            outputFile.write(f"\nANOVA Results -> F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
    # Perform Tukey's HSD test for pairwise comparisons
    # should I have an if and else? if p-value < 0.05, perform Tukey's HSD test?
    tukey_results = pairwise_tukeyhsd(df['sentiment_label'], df['Interest'])
    
    # Print Tukey's HSD results
    
    #print("\nTukey's HSD Test Results:")
    #print(tukey_results)
    if fileOutput == 1:
        with open(fileOutputName, "a") as outputFile:
            outputFile.write("\nTukey's HSD Test Results:")
            outputFile.write(tukey_results)
    # Perform Tukey's HSD test for pairwise comparisons
    # I want to extract most diffrent groups and visualize the results.
    #visualize_sentiment_analysis(df, 'sentiment_label', 'Interest', (p_value, p_value_loc2, p_value_loc3))

    return avg_sentiment_per_interest, results

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def t_test_between_states(df, state1, state2, LocationTtestOutput):
    state1_sentiment = df[df['Region'] == state1]['sentiment_label'].mean()
    state2_sentiment = df[df['Region'] == state2]['sentiment_label'].mean()
    t_stat, p_val = ttest_ind(state1_sentiment, state2_sentiment, equal_var=False)
    #print(f"t-test between {state1} and {state2}: t-statistic = {t_stat}, p-value = {p_val}")
  # Append the result to the CSV file
    result_df = pd.DataFrame([{
        'State 1': state1,
        'State 2': state2,
        't-statistic': t_stat,
        'p-value': p_val
    }])
    
    # Check if the file exists, if not create it with headers, otherwise append data
    try:
        result_df.to_csv(LocationTtestOutput, mode='a', header=False, index=False)
    except FileNotFoundError:
        result_df.to_csv(LocationTtestOutput, mode='w', header=True, index=False)

def analyze_Locations(df, LocationAnovaOutput, LocationTtestOutput):
    #assumption: Check Assumptions: ANOVA: groups have approximately normal distributions and similar variances.
    #Assumption: which specific (states) differ from each other.
        # use pairwise comparisons: Tukey's Honest Significant Difference (HSD).

    # 1. Group by each state
    # 2. Calculate the average sentiment score for each state
    # Perform ANOVA to confirm that there are overall significant differences.
    #Tukey's HSD: compares the means of every pair of (states) and identifies which pairs have significant differences.
   
    # Filter for US data
    us_df = df[df['Country'] == 'United States of America'].copy()
    state_mapping = {state: i+1 for i, state in enumerate(sorted(us_df['Region'].dropna().astype(str).unique()))}
    us_df['state_label'] = us_df['Region'].map(state_mapping)
    df['state_label'] = us_df['state_label']
    df.loc[:, 'state_label'].fillna(0, inplace=True)
    us_df.fillna(0, inplace=True)
    df.fillna(0, inplace=True)
    unique_states = us_df['Region'].unique()
    #t-test between states
    for i in range(len(unique_states)):
        for j in range(i + 1, len(unique_states)):  # Avoid redundant comparisons
            state1 = unique_states[i]
            state2 = unique_states[j]
            t_test_between_states(df, state1, state2, LocationTtestOutput)
    
# 3. Perform ANOVA: overall significant differences between states
    state_groups = [us_df[us_df['Region'] == state]['sentiment_label'] for state in us_df['Region'].unique()]
    f_statistic, p_value = stats.f_oneway(*state_groups)
    print(f"ANOVA F-statistic: {f_statistic}")
    print(f"ANOVA p-value: {p_value}")

    # 4. If ANOVA shows significant differences, proceed with Tukey's HSD for pairwise comparisons
    if p_value < 0.05:
        print("Significant differences found, performing Tukey's HSD test...")
        
        # Apply Tukey's HSD for pairwise comparison of states
        tukey = pairwise_tukeyhsd(endog=us_df['sentiment_label'],  # dependent variable (sentiment scores)
                                  groups=us_df['Region'],    # independent variable (states)
                                  alpha=0.05)                # significance level
              # Save Tukey's HSD results to the file
        tukey_results = []
        for comparison in tukey.summary().data[1:]:
            tukey_results.append({
                'State 1': comparison[0],
                'State 2': comparison[1],
                'Mean Difference': comparison[2],
                'p-value': comparison[3],
                'Lower CI': comparison[4],
                'Upper CI': comparison[5],
                'Significant': comparison[6]
            })
        
        tukey_results_df = pd.DataFrame(tukey_results)
        try:
            tukey_results_df.to_csv('TukeyHSDOutput.csv', mode='a', header=False, index=False)
        except FileNotFoundError:
            tukey_results_df.to_csv('TukeyHSDOutput.csv', mode='w', header=True, index=False)
        
        print(tukey)
        return tukey
    else:
        print("No significant differences found between states.")

        return None

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
 
def analyze_gender_differences(df):
        # Group by Gender and calculate mean sentiment scores
        gender_groups = df.groupby('Gender')['sentiment_label']
        means = gender_groups.mean()
        stds = gender_groups.std() #standard deviation; a measure of the amount of variation or dispersion in a set of values; uantifies how much the values in a dataset deviate from the mean (average) of the dataset.
        print("Mean Sentiment Scores by Gender:")
        print(means)
        print("\nStandard Deviation of Sentiment Scores by Gender:")
        print(stds)

        # Pairwise t-tests
        male_sentiments = df[df['Gender'] == 'male']['sentiment_label']
        female_sentiments = df[df['Gender'] == 'female']['sentiment_label']
        unknown_sentiments = df[df['Gender'] == 'unknown']['sentiment_label']
        t_stat_mf, p_value_mf = ttest_ind(male_sentiments, female_sentiments, equal_var=False)
        t_stat_mu, p_value_mu = ttest_ind(male_sentiments, unknown_sentiments, equal_var=False)
        t_stat_fu, p_value_fu = ttest_ind(female_sentiments, unknown_sentiments, equal_var=False)
        print("\nT-test Results:")
        print(f"Men vs Women: t-statistic = {t_stat_mf}, p-value = {p_value_mf}")
        print(f"Men vs Unknown: t-statistic = {t_stat_mu}, p-value = {p_value_mu}")
        print(f"Women vs Unknown: t-statistic = {t_stat_fu}, p-value = {p_value_fu}")

        # One-Way ANOVA
        #ols LR? is like?
        #Linear Regression (Ordinary Least Squares)
        model = ols(f'{'sentiment_label'} ~ C(Gender)', data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print("\nOne-Way ANOVA Results:")
        print(anova_table)

        # Post-hoc Test: Tukey's HSD
        tukey = pairwise_tukeyhsd(df['sentiment_label'], df['Gender'])
        print("\nTukey's HSD Post-hoc Test Results:")
        print(tukey)

        # Visualize with boxplots
        plot_sentiment_boxplots(df, 'sentiment_label')
        plot_mean_sentiment_bars(df, 'sentiment_label')
        plot_ttest_results(p_value_mf, p_value_mu, p_value_fu)
        plot_anova_residuals(model)
        plot_tukey_hsd(df,  'sentiment_label')
def plot_sentiment_boxplots(df, sentiment):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Gender', y=sentiment, data=df)
    plt.title(f'Boxplot of {sentiment.replace("_score", "").capitalize()} Sentiment by Gender')
    plt.ylabel(f'{sentiment.replace("_score", "").capitalize()} Sentiment Score')
    plt.xlabel('Gender')
    plt.show()

def plot_mean_sentiment_bars(df, sentiment):
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Gender', y=sentiment, data=df, errorbar=None)
        plt.title(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment by Gender')
        plt.ylabel(f'Mean {sentiment.replace("_score", "").capitalize()} Sentiment Score')
        plt.xlabel('Gender')
        plt.show()

def plot_ttest_results(p_value_mf, p_value_mu, p_value_fu):
    ttest_results = {
        "Comparison": ["Men vs Women", "Men vs Unknown", "Women vs Unknown"],
        "p-value": [p_value_mf, p_value_mu, p_value_fu]
    }
    ttest_df = pd.DataFrame(ttest_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Comparison", y="p-value", data=ttest_df)
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
    plt.title('Pairwise T-test Results')
    plt.ylabel('p-value')
    plt.legend()
    plt.show()

def plot_anova_residuals(model):
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.show()

def plot_tukey_hsd(df, sentiment):
    tukey = pairwise_tukeyhsd(df[sentiment], df['Gender'])
    tukey.plot_simultaneous()
    plt.title(f'Tukey HSD Test for {sentiment.replace("_score", "").capitalize()} Sentiment')
    plt.xlabel('Mean Difference')
    plt.show()

#for sentiment in ['positive_score', 'neutral_score', 'negative_score']:
   # plot_tukey_hsd(grouped_df, sentiment)
# Example of how to call the function:
# analyze_gender_differences(grouped_df)

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
'''
def calculate_sentiment_scores(df, model, tokenizer):
    positive_scores = []
    neutral_scores = []
    negative_scores = []  
    for row in range(len(df)):
        text = df["Snippet"].iloc[row]
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)  # [negative, neutral, positive]
        negative_scores.append(scores[0])
        neutral_scores.append(scores[1])
        positive_scores.append(scores[2])
    df['positive_score'] = positive_scores
    df['neutral_score'] = neutral_scores
    df['negative_score'] = negative_scores
    return df
'''
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
'''
def assign_sentiment_labels(df):
    # Assign labels based on the highest score
    conditions = [
        (df['positive_score'] > df['neutral_score']) & (df['positive_score'] > df['negative_score']),
        (df['neutral_score'] > df['positive_score']) & (df['neutral_score'] > df['negative_score']),
        (df['negative_score'] > df['positive_score']) & (df['negative_score'] > df['neutral_score'])
    ]
    choices = [1, 0, -1]
    df['sentiment_label'] = np.select(conditions, choices, default=0)
    return df

def calculate_average_sentiment_label(df):
    # Group by 'Author' and calculate the average sentiment label
    author_sentiment = df.groupby('Author')['sentiment_label'].mean().reset_index()
    author_sentiment.columns = ['Author', 'average_sentiment_label']
    return author_sentiment
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
'''
'''
def group_by_author(df):
    def most_common(x):
        value_counts = x.value_counts()
        if value_counts.empty:
            return None  # or some default value
        return value_counts.index[0]

    try:
        # Group by the 'Author' column
        grouped_df = df.groupby('Author').agg({
            'sentiment_label': 'mean',  # Avg
            'Gender': 'first',        # Assuming gender is the same for all posts by the same author
            'Country': 'first',       # Assuming country is the same for all posts by the same author
            'Region': 'first',        # Assuming region is the same for all posts by the same author
            'Engagement Type': most_common,
            'Interest': most_common
              # Handle empty cases
            #how can have mean of Interest?
            #'Average_Interest_Sentiment' = 'sentiment_label'.transform('mean')

        }).reset_index()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    return grouped_df

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def process_engagement_types(df, engagementTypeCounts):
    for tweet in range(len(df)):
        engagement_type = df.iloc[tweet]["Engagement Type"]
        if engagement_type == "RETWEET":
            engagementTypeCounts[0] += 1
        elif engagement_type == "QUOTE":
            engagementTypeCounts[1] += 1
        elif engagement_type == "REPLY":
            engagementTypeCounts[2] += 1
        else:
            engagementTypeCounts[3] += 1
    return engagementTypeCounts
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def extract_verified_users(df):
    #Extracts verified users from the DataFrame.
    verified_users = []
    for tweet in range(len(df)):
        verified = False
        if df.iloc[tweet]["Engagement Type"] == "RETWEET":
            # Extract original username from the retweet URL
            if pd.notna(df.iloc[tweet]["Twitter Retweet of"]):
                original_author = extract_username_from_url(df.iloc[tweet]["Twitter Retweet of"])
            print("Url: %s, original author: %s,  author column: %s" % (df.iloc[tweet]["Twitter Retweet of"], original_author, df.iloc[tweet]["Author"]))
            # Should we have another dataset or method to verify this original author's status?
            # verified = check_verified_status(original_author)
        elif pd.isna(df.iloc[tweet]["Engagement Type"]):
            original_author = df.iloc[tweet]["Author"]
            verified = df.iloc[tweet]["Twitter Verified"]
        # Append verified status and author
        if verified:
            verified_users.append(original_author)
    return verified_users
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def extract_username_from_url(url):
    # Extracts the username from a given Twitter URL
        url = url.lower()
        match = re.search(r'twitter\.com/([^/]+)/status', url)
        if match:
            return match.group(1)
        return None
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
'''
def main():
    # Load the data
    df = pd.read_pickle(sys.argv[1])
    tweetsLimit = int(sys.argv[2])
    #fileOutput = int(sys.argv[3])  # if 1, output results to file, else output results to stdio
    #fileOutputName = sys.argv[4]
    tweetsLimit = min(tweetsLimit, len(df.index))
   #if fileOutput == 1:
        #outputFile = open(fileOutputName, "w")

    # Process DataFrame
    df = df.head(tweetsLimit)
    #df = df.dropna(subset=['Gender', 'Country', 'Region'])
    #engagementTypeCounts = [0, 0, 0, 0] # number of retweets, quotes, replies, original posts
    #engagementTypeCounts = process_engagement_types(df, engagementTypeCounts)
    #verified_users = extract_verified_users(df)
    #print(verified_users)
    # Encode Gender
    #df_encoded = pd.get_dummies(df['Gender'], prefix='Gender', drop_first=True)
    # Load the sentiment model
   ### MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ###tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ###model = AutoModelForSequenceClassification.from_pretrained(MODEL)
   ### df = calculate_sentiment_scores(df, model, tokenizer)
  ###  df = assign_sentiment_labels(df)
   ### grouped_by_author_df = group_by_author(df)
process ed_df, grouped_by_author_df = preprocess_data(df)

    #calculate_average_sentiment_label(df)
    LocationAnovaOutput = "output/LocationAnovaOutput"
    LocationTtestOutput = "output/LocationTtestOutput"
    #analyze_Locations(grouped_by_author_df, LocationAnovaOutput, LocationTtestOutput)
    AccountTypeTtestOutput = "output/output/AccountTypeTtestOutput"
    AccountTypeAnovaOutput = "output/output/AccountTypeAnovaOutput"
    RegionalTtestOutput = "output/RegionalTtestOutput"
    RegionalAnovaOutput = "output/RegionalAnovaOutput"
    Regional_analysis(grouped_by_author_df, RegionalTtestOutput, RegionalAnovaOutput)
    Verifiedـusers(grouped_by_author_df, AccountTypeTtestOutput, AccountTypeAnovaOutput)
    #analyze_interest_users(grouped_by_author_df, fileOutput, fileOutputName)
    return
    analyze_gender_differences(grouped_by_author_df)
    analyze_Locations(grouped_by_author_df, LocationAnovaOutput, LocationTtestOutput)

    

    # Filter for US data
    us_df = df[df['Country'] == 'United States of America']
    state_mapping = {state: i+1 for i, state in enumerate(sorted(us_df['Region'].dropna().astype(str).unique()))}
    us_df['state_label'] = us_df['Region'].map(state_mapping)
    df['state_label'] = us_df['state_label']
    df.loc[:, 'state_label'].fillna(0, inplace=True)
    us_df.fillna(0, inplace=True)
    df.fillna(0, inplace=True)
   

    
  
   
    # Prepare data for modeling
    X = df_encoded.reset_index(drop=True)
    y = df['sentiment_score'].reset_index(drop=True)

    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Fit the model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    X_df = pd.DataFrame(X, columns=df_encoded.columns)
    coefficients = pd.DataFrame(regressor.coef_, X_df.columns, columns=['Coefficient'])

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(coefficients)

    # Display results
    test_results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    print(test_results.head())

    # Statistical analysis
    male_sentiments = df[df['Gender'] == 'male']['sentiment_score']
    female_sentiments = df[df['Gender'] == 'female']['sentiment_score']
    t_stat, p_value = ttest_ind(male_sentiments, female_sentiments, equal_var=False)
    print(f'T-test statistic: {t_stat}')
    print(f'P-value: {p_value}')

    male_encoded = df_encoded.loc[male_sentiments.index].reset_index(drop=True)
    male_predictions = regressor.predict(male_encoded)
    df.loc[male_sentiments.index, 'predicted_sentiment_score'] = male_predictions
   
    # Ensure 'Gender' is categorical
    df['Gender'] = df['Gender'].astype('category')

    # Two-Way ANOVA
    model = ols('sentiment_score ~ C(Gender) + C(state_label) + C(Gender):C(state_label)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Output results to file or stdio
    if fileOutput == 1:
        outputFile.write(f"Mean Squared Error: {mse}\n")
        outputFile.write(f"R^2 Score: {r2}\n")
        coefficients.to_csv(outputFile, index=True)
        test_results.to_csv(outputFile, index=False)
        df[['Snippet', 'predicted_sentiment_score']].to_csv(outputFile, index=False)
        outputFile.write(f"T-test Statistic: {t_stat}\n")
        outputFile.write(f"P-value: {p_value}\n")
        outputFile.write(f"ANOVA Table:\n{anova_table}\n")
        outputFile.close()

if __name__ == "__main__":
    main()
