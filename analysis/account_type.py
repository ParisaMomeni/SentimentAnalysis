
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
#______________________________________________________________________________________
processed_df = pd.read_pickle('data/processed_data.pkl')
df = processed_df
df = df.dropna(subset=['Account Type'])
    
start_date = "2022-01-01"
end_date = "2022-12-31"
bin_edges = pd.date_range(start=start_date, end="2023-01-01", freq="2MS")
labels = [
        "Jan-Feb", "Mar-Apr", "May-Jun", "Jul-Aug", "Sep-Oct", "Nov-Dec"
    ]
    
# Assign each date to a bimonthly period
df['Bimonthly'] = pd.cut(
    df['Date'],
    bins=bin_edges,
    labels=labels,
    right=False
    )
    
# Group data by Account Type and Bimonthly periods, calculate the mean sentiment
grouped = df.groupby(['Account Type', 'Bimonthly'])['sentiment_label'].mean().reset_index()

# Plot the changes in sentiment over bimonthly periods for each Account Type
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=grouped,
    x="Bimonthly",
    y="sentiment_label",
    hue="Account Type",
    marker="o",
    palette="Set2"
    )

 # Customize the plot
plt.title("Changes in Sentiment Over Bimonthly Periods by Account Type", fontsize=16, fontweight="bold")
plt.xlabel("Bimonthly Period")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.legend(title="Account Type")
plt.grid(visible=True, linestyle="--", alpha=0.7)

# Save the plot
output_root = "output/account_type"
if not os.path.exists(output_root):
    os.makedirs(output_root)
plt.savefig(f"{output_root}/sentiment_line_chart_by_account_type.png")
plt.show()

account_type_stats = df.groupby('Account Type')['sentiment_label'].agg(['mean', 'std']).reset_index()
account_type_stats.columns = ['Account_Type', 'Mean_Sentiment', 'Std_Sentiment']

account_type_stats = account_type_stats.sort_values('Mean_Sentiment', ascending=False)

output_file = os.path.join(output_root, 'account_type_sentiment_stats.csv')
account_type_stats.to_csv(output_file, index=False, header=True)

print(f"Account type sentiment statistics have been written to {output_file}")
print("\nAccount Type Sentiment Statistics:")
print(account_type_stats)

