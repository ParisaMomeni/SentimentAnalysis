import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the descriptive statistics
file_path_descriptive = "output/gender/gender_descriptive_stats.csv"
descriptive_df = pd.read_csv(file_path_descriptive)

print("Data shape:", descriptive_df.shape)
print("Columns:", descriptive_df.columns)
print("First few rows:")
print(descriptive_df.head())

# Check if 'Gender' is a column or an index
if 'Gender' not in descriptive_df.columns and descriptive_df.index.name == 'Gender':
    descriptive_df = descriptive_df.reset_index()

# Check that the necessary columns exist
if 'Gender' in descriptive_df.columns and 'mean' in descriptive_df.columns and 'std' in descriptive_df.columns:
    # 3. Bar plot of mean sentiments with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(descriptive_df['Gender'], descriptive_df['mean'], 
                   yerr=descriptive_df['std'], capsize=5, 
                   color=[ 'pink', 'skyblue', 'lightgreen'])
    
    plt.title('Mean Sentiment by Gender', fontsize=18, fontweight='bold')
    plt.ylabel('Mean Sentiment', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("output/gender/gender_mean_sentiment_barplot.png")

    plt.figure(figsize=(10, 6))
sns.boxplot(data=descriptive_df, x='Gender', y='mean', palette=['pink', 'skyblue', 'lightgreen'])
plt.title('Sentiment Score Distribution by Gender', fontsize=18, fontweight='bold')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Sentiment Score', fontsize=14)
plt.tight_layout()
plt.savefig("output/gender/gender_mean_boxplot.png")

plt.figure(figsize=(10, 6))
for gender in descriptive_df['Gender'].unique():
    subset = descriptive_df[descriptive_df['Gender'] == gender]
    sns.kdeplot(subset['mean'], label=gender, fill=True, alpha=0.4)

plt.title('Sentiment Score Distribution by Gender', fontsize=18, fontweight='bold')
plt.xlabel('Sentiment Score', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig("output/gender/gender_mean_distribution_plot.png")
plt.close()
