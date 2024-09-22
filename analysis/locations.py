import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def analyze_Locations(df, output_folder):
    # Ensure the output folder exists
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Calculate statistics for each country
    country_stats = df.groupby('Country')['sentiment_label'].agg(['mean', 'std', 'count'])
    country_stats = country_stats.sort_values('mean', ascending=False)

    # Save country statistics
    country_stats.to_csv(f"{output_folder}/country_statistics.csv")

    # Perform one-way ANOVA
    model = ols('sentiment_label ~ C(Country)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"{output_folder}/country_anova_results.csv")

    # Perform Tukey's HSD test for pairwise comparisons
    tukey_results = pairwise_tukeyhsd(df['sentiment_label'], df['Country'])
    with open(f"{output_folder}/country_tukey_hsd_results.txt", 'w') as f:
        f.write(str(tukey_results))

    # Perform t-tests between all pairs of countries
    countries = df['Country'].unique()
    with open(f"{output_folder}/country_pairwise_ttests.txt", 'w') as f:
        for i in range(len(countries)):
            for j in range(i+1, len(countries)):
                country1 = df[df['Country'] == countries[i]]['sentiment_label']
                country2 = df[df['Country'] == countries[j]]['sentiment_label']
                t_stat, p_val = stats.ttest_ind(country1, country2)
                f.write(f"{countries[i]} vs {countries[j]}: t-statistic = {t_stat}, p-value = {p_val}\n")

    # Visualizations

    # 1. Bar plot of mean sentiments by country
    plt.figure(figsize=(12, 6))
    country_stats['mean'].plot(kind='bar', yerr=country_stats['std'], capsize=5)
    plt.title('Mean Sentiment by Country')
    plt.xlabel('Country')
    plt.ylabel('Mean Sentiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/country_mean_sentiment_barplot.png")
    plt.close()

    # 2. Box plot of sentiments by country
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Country', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Country')
    plt.xlabel('Country')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/country_sentiment_boxplot.png")
    plt.close()

    # 3. Heatmap of sentiment correlation between countries
    pivot_df = df.pivot_table(values='sentiment_label', index='Author', columns='Country', aggfunc='first')
    correlation_matrix = pivot_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation of Sentiments Between Countries')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/country_correlation_heatmap.png")
    plt.close()

    # 4. Violin plot of sentiments by country
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Country', y='sentiment_label', data=df)
    plt.title('Sentiment Distribution by Country (Violin Plot)')
    plt.xlabel('Country')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/country_sentiment_violinplot.png")
    plt.close()

    # 5. Scatter plot of sentiment vs. number of users for each country
    plt.figure(figsize=(12, 8))
    plt.scatter(country_stats['count'], country_stats['mean'], alpha=0.6)
    for idx, row in country_stats.iterrows():
        plt.annotate(idx, (row['count'], row['mean']))
    plt.title('Mean Sentiment vs. Number of Users by Country')
    plt.xlabel('Number of Users')
    plt.ylabel('Mean Sentiment')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/country_sentiment_vs_users_scatter.png")
    plt.close()

    # 6. World map of mean sentiments (if you have country codes)
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = world.merge(country_stats, how='left', left_on=['name'], right_on=['Country'])
        plt.figure(figsize=(20, 10))
        world.plot(column='mean', cmap='RdYlGn', linewidth=0.8, edgecolor='0.8', legend=True, 
                   missing_kwds={'color': 'lightgrey'})
        plt.title('Mean Sentiment by Country')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/world_map_sentiment.png")
        plt.close()
    except ImportError:
        print("geopandas not installed. Skipping world map visualization.")

    print(f"Location analysis complete. Results saved in {output_folder}")

if __name__ == "__main__":
    # This allows you to run this analysis independently if needed
    import sys
    if len(sys.argv) != 3:
        print("Usage: python locations.py <grouped_data_file> <output_folder>")
        sys.exit(1)

    grouped_data_file = sys.argv[1]
    output_folder = sys.argv[2]

    print(f"Loading grouped data from {grouped_data_file}...")
    grouped_by_author_df = pd.read_pickle(grouped_data_file)

    analyze_Locations(grouped_by_author_df, output_folder)