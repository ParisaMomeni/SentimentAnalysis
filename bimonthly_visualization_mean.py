import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

processed_df = pd.read_pickle('data/processed_data.pkl')
df = processed_df.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
bin_edges = pd.date_range(start="2021-01-01", end="2024-01-01", freq="2MS")
labels = [f"{bin_edges[i].strftime('%b-%Y')}-{bin_edges[i + 1].strftime('%b-%Y')}" for i in range(len(bin_edges) - 1)]
df['Bimonthly'] = pd.cut(df['Date'], bins=bin_edges, labels=labels, right=False)
grouped_gender = df.groupby(['Gender', 'Bimonthly'])['sentiment_label'].mean().reset_index()

sentimentByInterestTemporal = {}
for _, row in df.iterrows():
    if not pd.isna(row["Interest"]):
        sentiment_value = int(row["sentiment_label"])  # -1, 0, or 1
        bimonthly_period = row["Bimonthly"]
        for interest in row["Interest"].split(","):
            interest = interest.strip()
            if interest not in sentimentByInterestTemporal:
                sentimentByInterestTemporal[interest] = {}
            if bimonthly_period not in sentimentByInterestTemporal[interest]:
                sentimentByInterestTemporal[interest][bimonthly_period] = {"negative": 0, "neutral": 0, "positive": 0}
            if sentiment_value == -1:
                sentimentByInterestTemporal[interest][bimonthly_period]["negative"] += 1
            elif sentiment_value == 0:
                sentimentByInterestTemporal[interest][bimonthly_period]["neutral"] += 1
            elif sentiment_value == 1:
                sentimentByInterestTemporal[interest][bimonthly_period]["positive"] += 1

temporal_data = []
for interest, periods in sentimentByInterestTemporal.items():
    for period, counts in periods.items():
        total = sum(counts.values())
        if total > 0:
            mean_sentiment = (counts["positive"] - counts["negative"]) / total
            temporal_data.append({
                "Interest": interest,
                "Bimonthly": period,
                "Mean Sentiment": mean_sentiment,
                "Count": total
                })

temporal_df = pd.DataFrame(temporal_data)
interestsOutputRoot = "output/interests"
os.makedirs(interestsOutputRoot, exist_ok=True)
temporal_df.to_csv(f"{interestsOutputRoot}/interests_temporal.csv", index=False)

# Adding verified vs non-verified sentiment calculation
df['Twitter Verified'] = df['Twitter Verified'].fillna(False)  # Ensure no missing data for verified status
grouped_verified = df.groupby(['Twitter Verified', 'Bimonthly'])['sentiment_label'].mean().reset_index()

#Group by Account Type and Bimonthly period to calculate mean sentiment
df['Account Type'] = df['Account Type'].fillna('Unknown')  # Fill with a default value
grouped_account = df.groupby(['Account Type', 'Bimonthly'])['sentiment_label'].mean().reset_index()

#Regional 
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
df['state_to_region'] = df['Region'].map(state_to_region)
df['state_to_region'] = df['state_to_region'].fillna('Unknown')
grouped_region = df.groupby(['state_to_region', 'Bimonthly'])['sentiment_label'].mean().reset_index()
print(grouped_region.head())
grouped_region = grouped_region.dropna(subset=['sentiment_label'])

#grouped_region['sentiment_label'] = grouped_region['sentiment_label'].fillna(0)

# Plotting
fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

# Plot 1: Gender-based analysis
sns.lineplot(ax=axes[0], data=grouped_gender, x='Bimonthly', y='sentiment_label', hue='Gender', marker='o', palette='Set2')
axes[0].set_title("Bimonthly Sentiment Analysis by Gender", fontsize=16)
axes[0].set_xlabel("Bimonthly Period", fontsize=14)
axes[0].set_ylabel("Average Sentiment Score", fontsize=14)
axes[0].legend(title='Gender')
axes[0].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 2: Interest-based analysis (choose top 5 interests for clarity)
top_interests = temporal_df.groupby('Interest')['Mean Sentiment'].mean().nlargest(5).index
sns.lineplot(ax=axes[1], data=temporal_df[temporal_df['Interest'].isin(top_interests)], 
             x='Bimonthly', y='Mean Sentiment', hue='Interest', marker='o', palette='tab10')
axes[1].set_title("Bimonthly Sentiment Analysis by Interest", fontsize=16)
axes[1].set_xlabel("Bimonthly Period", fontsize=14)
axes[1].set_ylabel("Average Sentiment Score", fontsize=14)
axes[1].legend(title='Interest')
axes[1].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 3: Verified vs Non-Verified Users Analysis
sns.lineplot(ax=axes[2], data=grouped_verified, x='Bimonthly', y='sentiment_label', hue='Twitter Verified', marker='o', palette='coolwarm')
axes[2].set_title("Bimonthly Sentiment Analysis by Verified Status", fontsize=16)
axes[2].set_xlabel("Bimonthly Period", fontsize=14)
axes[2].set_ylabel("Average Sentiment Score", fontsize=14)
axes[2].legend(title='Twitter Verified')
axes[2].grid(visible=True, linestyle='--', alpha=0.6)
# Plot 4: Account Type Temporal Sentiment Analysis
sns.lineplot(ax=axes[3], data=grouped_account, x='Bimonthly', y='sentiment_label', hue='Account Type', marker='o', palette='Set2')
axes[3].set_title("Bimonthly Sentiment Analysis by Account Type", fontsize=16)
axes[3].set_xlabel("Bimonthly Period", fontsize=14)
axes[3].set_ylabel("Average Sentiment Score", fontsize=14)
axes[3].legend(title='Account Type')
axes[3].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 5: Account Type Temporal Sentiment Analysis
sns.lineplot(ax=axes[4], data=grouped_region, x='Bimonthly', y='sentiment_label', hue='state_to_region', marker='o', palette='Set2')
axes[4].set_title("Bimonthly Sentiment Analysis by Regional", fontsize=16)
axes[4].set_xlabel("Bimonthly Period", fontsize=14)
axes[4].set_ylabel("Average Sentiment Score", fontsize=14)
axes[4].legend(title='Region')
axes[4].grid(visible=True, linestyle='--', alpha=0.6)

plt.xticks(rotation=45)
# Save the combined plot
output_dir = "output/bimonthly/"
os.makedirs(output_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{output_dir}bimonthly_gender_interest_sentiment_analysis2.png")
plt.show()
