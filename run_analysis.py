#I will put another argument for independent variable and a switch case #if else
import pandas as pd
import sys
import os

from analysis.regional import Regional_analysis
from analysis.verified_users import Verified_users
from analysis.interests import analyze_interest_users
from analysis.gender import analyze_gender_differences
from analysis.locations import analyze_Locations

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <tweets_limit>")
        sys.exit(1)

   # processed_data_file = sys.argv[1]
  #  grouped_data_file = sys.argv[2]
    tweetsLimit = int(sys.argv[1])

    # Load preprocessed data
    print(f"Loading processed data from {data/processed_data_file}...")
    processed_df = pd.read_pickle('data/processed_data.pkl')
    
    print(f"Loading grouped data from {data/grouped_data_file}...")
    grouped_by_author_df = pd.read_pickle('data/grouped_data.pkl')

    # Apply tweets limit
    processed_df = processed_df.head(tweetsLimit)
    grouped_by_author_df = grouped_by_author_df.head(tweetsLimit)


    # Perform various analyses
    print("Performing Regional analysis...")
    Regional_analysis(grouped_by_author_df)
    '''
    print("Analyzing Verified users...")
    Verified_users(processed_df, "output/verified_users")
    
    print("Analyzing interests...")
    analyze_interest_users(grouped_by_author_df, "output/interests")
    
    print("Analyzing gender differences...")
    analyze_gender_differences(grouped_by_author_df, "output/gender")
    
    print("Analyzing Locations...")
    analyze_Locations(grouped_by_author_df, "output/locations")

    print("Analysis complete!")
'''
if __name__ == "__main__":
    main()