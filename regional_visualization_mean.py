import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the regional descriptive statistics
file_path_descriptive = "output/regional/regional_mean.csv"
descriptive_df = pd.read_csv(file_path_descriptive)

# Check that the necessary columns exist
if 'mean' in descriptive_df.columns and 'std' in descriptive_df.columns and 'state_to_region' in descriptive_df.columns:
    # Extract data for plotting
    regions = descriptive_df['state_to_region']
    mean_sentiments = descriptive_df['mean']
    std_deviations = descriptive_df['std']

    # First plot: Colorful bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(regions, mean_sentiments, yerr=std_deviations, capsize=5, 
                   color=[ 'pink', 'skyblue', 'lightgreen', 'salmon','lemonchiffon'])   

    plt.title('Mean Sentiment by Region', fontsize=18, fontweight='bold')
    plt.ylabel('Mean Sentiment', fontsize=14)
   # plt.ylim(0, 1.0)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(range(len(regions)), regions, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig("output/regional/regional_mean_sentiment_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Colorful regional bar plot created and saved successfully.")

    