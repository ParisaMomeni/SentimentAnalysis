import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('output/account_type/account_type_sentiment_stats.csv')

# Set the style to match other visualizations
plt.style.use('default')
sns.set_palette("deep")

# Create a bar plot with blue bars
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Account_Type'], df['Mean_Sentiment'], yerr=df['Std_Sentiment'], capsize=5, color='blue', alpha=0.7)

# Customize the plot
plt.title('Mean Sentiment by Account Type', fontsize=16)
plt.xlabel('Account Type', fontsize=12)
plt.ylabel('Mean Sentiment', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom' if height > 0 else 'top',
             fontsize=10)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('output/account_type/account_type_sentiment_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as 'output/account_type/account_type_sentiment_plot.png'")

# Create a summary table
summary = df.describe()
summary.to_csv('output/account_type/account_type_summary.csv')
print("Summary statistics saved as 'output/account_type/account_type_summary.csv'")

# Print the data
print("\nAccount Type Sentiment Statistics:")
print(df)