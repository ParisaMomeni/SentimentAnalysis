

import pandas as pd
import numpy as np
import os

def analyze_account_types(df):
    output_root = "output/account_type"

    df = df.dropna(subset=['Account Type'])
    # Group by Account Type and calculate mean and std of sentiment
    account_type_stats = df.groupby('Account Type')['sentiment_label'].agg(['mean', 'std']).reset_index()
    # Rename columns for clarity
    account_type_stats.columns = ['Account_Type', 'Mean_Sentiment', 'Std_Sentiment']


    # Sort by Mean_Sentiment
    account_type_stats = account_type_stats.sort_values('Mean_Sentiment', ascending=False)

    # Write results to CSV file
    output_file = os.path.join(output_root, 'account_type_sentiment_stats.csv')
    account_type_stats.to_csv(output_file, index=False, header=True)

    print(f"Account type sentiment statistics have been written to {output_file}")

    # Print the results
    print("\nAccount Type Sentiment Statistics:")
    print(account_type_stats)

    return account_type_stats
