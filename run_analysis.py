#I will put another argument for independent variable and a switch case #if else
import pandas as pd
import sys
import os

from analysis.regional import Regional_analysis
from analysis.verified_users import Verified_users
from analysis.interests import analyze_interest_users
from analysis.gender import analyze_gender_differences
from analysis.locations import analyze_Locations
from analysis.account_type import analyze_account_types
from analysis.activity import analyze_author_activity

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <analysis_type>\n 1: regional/ 2: gender/ 3: location/ 4: verified/ 5: interests")       
        sys.exit(1)
        
    analysis_type = int(sys.argv[1])

    # Load preprocessed data
    print(f"Loading processed data from {'data/processed_data_file'}...")
    processed_df = pd.read_pickle('data/processed_data.pkl')
    
    print(f"Loading grouped data from {'data/grouped_data_file'}...")
    grouped_by_author_df = pd.read_pickle('data/grouped_data.pkl')

    # Apply tweets limit
    #processed_df = processed_df.head(tweetsLimit)
    #grouped_by_author_df = grouped_by_author_df.head(tweetsLimit)


    # Perform various analyses
    match (analysis_type):
        case 1:
            print("Performing Regional analysis...")
            Regional_analysis(grouped_by_author_df)
        case 2:
            print("Analyzing Verified users...")
            Verified_users(processed_df)
        case 3:
            print("Analyzing interests...")
            analyze_interest_users(grouped_by_author_df)
        case 4:
            print("Analyzing gender differences...")
            analyze_gender_differences(grouped_by_author_df)
        case 5:
            print("Analyzing Locations...")
            analyze_Locations(grouped_by_author_df)
        case 6:
            print("Analyzing Account type...")
            analyze_account_types(processed_df)
        case 7:
            print("Analyzing Author Activity...")
            analyze_author_activity(processed_df)
   

if __name__ == "__main__":
    main()