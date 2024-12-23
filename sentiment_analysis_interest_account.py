import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

processed_df = pd.read_pickle('data/processed_data.pkl')
df = processed_df.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
bin_edges = pd.date_range(start="2021-01-01", end="2024-01-01", freq="2MS")
labels = [f"{bin_edges[i].strftime('%b-%Y')}-{bin_edges[i + 1].strftime('%b-%Y')}" for i in range(len(bin_edges) - 1)]
df['Bimonthly'] = pd.cut(df['Date'], bins=bin_edges, labels=labels, right=False)

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
#_______________________________________________________________________________________________________________________
# Calculate the total popularity for each interest
interest_popularity = {}

for interest, periods in sentimentByInterestTemporal.items():
    total_count = sum(
        sum(period_data.values()) for period_data in periods.values()
    )
    interest_popularity[interest] = total_count

# Convert to DataFrame for easier handling
popularity_df = pd.DataFrame(
    list(interest_popularity.items()), columns=["Interest", "Count"]
)

# Sort by popularity
popularity_df = popularity_df.sort_values(by="Count", ascending=False)


# Plot the top 20 most popular interests
top_popular_interests = popularity_df.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_popular_interests, x="Count", y="Interest", palette="viridis")
plt.title("Top 20 Most Popular Interests")
plt.xlabel("Numbers")
plt.ylabel("Interest")
plt.tight_layout()
plt.savefig('output/interest_account/Top_20_Interests.png', dpi=300, bbox_inches='tight')
plt.show()


# Print or save the table
print("Popularity Summary (Top 20):")
print(popularity_df.head(20).to_string(index=False))

# Save to CSV for further analysis
popularity_df.to_csv('output/interest_account/popularity_summary.csv', index=False)

#_______________________________________________________________________________________________________________________
df['Account Type'] = df['Account Type'].fillna('Unknown') 
grouped_account = df.groupby(['Account Type', 'Bimonthly'])['sentiment_label'].mean().reset_index()
top_interests = temporal_df.groupby('Interest')['Mean Sentiment'].mean().nlargest(20).index

#ANOVA


#Encode categorical features
df_ml = df[['sentiment_label', 'Interest', 'Account Type']].dropna()
df_ml['Interest'] = LabelEncoder().fit_transform(df_ml['Interest'])
df_ml['Account Type'] = LabelEncoder().fit_transform(df_ml['Account Type'])
# Split data
X = df_ml[['Interest', 'Account Type']]
y = df_ml['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

#The variable top_interests contain the names of the top x interests with the highest average sentiment scores, sorted from highest to lowest sentiment.
x = 20
top_interests = temporal_df.groupby('Interest')['Mean Sentiment'].mean().nlargest(x).index
top_interest_df = df[df['Interest'].isin(top_interests)]

#Calculate mean sentiments and differences for annotations
sentiment_means = top_interest_df.groupby(['Interest', 'Account Type'])['sentiment_label'].mean().unstack()
sentiment_diff = abs(sentiment_means['individual'] - sentiment_means['organisational'])
max_diff_interest = sentiment_diff.idxmax()

# Create the bar chart
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=top_interest_df,
    x="Interest", 
    y="sentiment_label", 
    hue="Account Type", 
    estimator="mean",
    ci=None
)

# Add title and labels with the largest difference information
plt.title(f"Average Sentiment by Account Type and Top 20 Interests\nLargest Difference: {max_diff_interest} ({sentiment_diff[max_diff_interest]:.3f})")
plt.xlabel("Interest")
plt.ylabel("Average Sentiment")
plt.legend(title="Account Type")
plt.xticks(rotation=45)

# Add difference annotations above each pair of bars
x_coords = []
for i in range(len(sentiment_means.index)):
    x_coords.append(i)

y_max = top_interest_df.groupby('Interest')['sentiment_label'].max()
for idx, interest in enumerate(sentiment_means.index):
    diff = sentiment_diff[interest]
    ind_sent = sentiment_means.loc[interest, 'individual']
    org_sent = sentiment_means.loc[interest, 'organisational']
    
    # Add difference annotation
    plt.annotate(
        f'Δ = {diff:.3f}', 
        xy=(idx, max(ind_sent, org_sent)),
        xytext=(0, -150),
        textcoords='offset points',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        fontsize=9
    )
    
plt.margins(y=0.2)  
plt.tight_layout()

# Save the figure
plt.savefig('output/interest_account/BarChart.png', dpi=300, bbox_inches='tight')
plt.show()

# Print numerical summary
print("\nSentiment Differences Summary:")
summary_df = pd.DataFrame({
    'Individual': sentiment_means['individual'],
    'Organizational': sentiment_means['organisational'],
    'Absolute_Difference': sentiment_diff
}).round(3)
summary_df = summary_df.sort_values('Absolute_Difference', ascending=False)
print(summary_df)

#_______________________________________________________________________________________________________________________
# Heatmap for Sentiment Scores
heatmap_data = df[df['Interest'].isin(top_interests)].pivot_table(
    index="Interest",
    columns="Account Type",
    values="sentiment_label",
    aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mean Sentiment Heatmap by Account Type and Top Interests")
plt.xlabel("Account Type")
plt.ylabel("Interest")
plt.tight_layout()
plt.show()
plt.savefig('output/interest_account/Heatmap.png', dpi=300, bbox_inches='tight')

#_______________________________________________________________________________________________________________________

# Scatter plot with regression line




