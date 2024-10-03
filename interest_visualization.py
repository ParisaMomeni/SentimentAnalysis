import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('output/interests/interests_stats.csv')

# Sort the DataFrame by mean sentiment
df_sorted = df.sort_values('mean')

# Create the plot
plt.figure(figsize=(15, 10))
bars = plt.bar(df_sorted['interest'], df_sorted['mean'], yerr=df_sorted['std'], capsize=5)

# Customize the plot
plt.title('Mean Sentiment by Interest Category with Standard Deviation', fontsize=16)
plt.xlabel('Interest Category', fontsize=12)
plt.ylabel('Mean Sentiment', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom' if height < 0 else 'top',
             rotation=0, fontsize=8)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('output/interests/sentiment_by_interest.png')