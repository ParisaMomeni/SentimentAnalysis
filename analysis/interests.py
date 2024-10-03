import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os

def analyze_interest_users(df):
    interestsOutputRoot = f"output/interests/interests"
 
    sentimentByInterest = {} # "interest": score


   # Measure sentiment by user interest
   # if not pd.isna(df["Interest"]['sentiment_label']) :
        #for interest in df["Interest"]['sentiment_label'].split(",") :
        #    interest = interest.strip()
           # if not interest in sentimentByInterest :
             #   sentimentByInterest[interest] = [0, 0, 0]
                #sentimentByInterest[interest][index] += 1

    for _, row in df.iterrows():
        if not pd.isna(row["Interest"]):
            sentiment_value = int(row["sentiment_label"])  # -1, 0, or 1
            for interest in row["Interest"].split(","):
                interest = interest.strip()
                if interest not in sentimentByInterest:
                    sentimentByInterest[interest] = {"negative": 0, "neutral": 0, "positive": 0}
                if sentiment_value == -1:
                    sentimentByInterest[interest]["negative"] += 1
                elif sentiment_value == 0:
                    sentimentByInterest[interest]["neutral"] += 1
                elif sentiment_value == 1:
                    sentimentByInterest[interest]["positive"] += 1

    # Calculate statistics for each interest
    interest_stats = {}
    for interest, counts in sentimentByInterest.items():
        total = sum(counts.values())
        mean = (counts['positive'] - counts['negative']) / total
    
        # Calculate standard deviation
        values = [-1] * counts['negative'] + [0] * counts['neutral'] + [1] * counts['positive']
        std = np.std(values) if len(values) > 1 else 0
    
        interest_stats[interest] = {
            'mean': mean,
            'std': std,
        'count': total
    }

       # Convert interest_stats to a DataFrame and save as CSV
    interest_stats_df = pd.DataFrame.from_dict(interest_stats, orient='index')
    interest_stats_df.index.name = 'interest'
    interest_stats_df.to_csv(f"{interestsOutputRoot}_stats.csv")
 
    