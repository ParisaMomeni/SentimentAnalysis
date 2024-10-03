import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
import numpy as np
import plotly.graph_objects as go

# 1. Box plot
df = pd.read_csv('output/gender/gender_descriptive_stats.csv')
print(df.head())
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='mean', data=df)
plt.title('Sentiment Distribution by Gender')
plt.savefig(f"output/gender/sentiment_boxplot.png")
plt.close()
plt.show()
# 2. Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='mean', data=df)
plt.title('Sentiment Distribution by Gender (Violin Plot)')
plt.savefig(f"output/gender/sentiment_violinplot.png")
plt.close()
plt.show()


# Load the descriptive statistics
file_path_descriptive = "output/gender/gender_descriptive_stats.csv"
descriptive_df = pd.read_csv(file_path_descriptive)

# Check that the necessary columns exist
if 'mean' in descriptive_df.columns and 'std' in descriptive_df.columns:
    # 3. Bar plot of mean sentiments with error bars
    plt.figure(figsize=(10, 6))
    descriptive_df['mean'].plot(kind='bar', yerr=descriptive_df['std'], capsize=5)
    plt.title('Mean Sentiment by Gender')
    plt.ylabel('Mean Sentiment')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/gender/gender_mean_sentiment_barplot.png")
    plt.close()
else:
    print("Required columns ('mean', 'std') are missing from the descriptive statistics file.")

# 4. Histogram of sentiments for each gender
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='mean', hue='Gender', element='step', stat='density', common_norm=False)
plt.title('Distribution of Sentiments by Gender')
plt.xlabel('Sentiment Score')
plt.ylabel('Density')
plt.savefig(f"output/gender/gender_sentiment_histogram.png")
plt.close()

print(f"Gender analysis complete. Results saved in output/gender")
plt.show()
#----------------------------------------------------------
# Load the ANOVA results from the CSV file
anova_file_path = "output/gender/gender_anova_results.csv"
anova_df = pd.read_csv(anova_file_path)
print(anova_df.columns)
# Set up the bar plot for sum of squares
plt.figure(figsize=(12, 6))

# Plot for Sum of Squares
plt.subplot(1, 2, 1)
plt.bar(anova_df['source'], anova_df['sum_sq'], color=['#1f77b4', '#ff7f0e'])
plt.title('Sum of Squares by Group')
plt.ylabel('Sum of Squares')
plt.xlabel('source')
plt.xticks(rotation=0)

# Plot for F-statistic
plt.subplot(1, 2, 2)
plt.bar(anova_df['source'], anova_df['F'], color=['#1f77b4', '#ff7f0e'])
plt.title('F-statistic by Group')
plt.ylabel('F-statistic')
plt.xlabel('Source')
plt.xticks(rotation=0)

plt.tight_layout()

# Save the plots as a PNG file
plt.savefig("output/gender/anova_results.png")
plt.show()

# 5. Pie chart of gender distribution
plt.figure(figsize=(8, 8))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.savefig("output/gender/gender_distribution_pie.png")
plt.close()

# 6. Stacked bar chart of sentiment categories by gender
plt.figure(figsize=(10, 6))
sentiment_categories = pd.cut(df['mean'], bins=[-1, -0.33, 0.33, 1], labels=['Negative', 'Neutral', 'Positive'])
sentiment_by_gender = pd.crosstab(df['Gender'], sentiment_categories, normalize='index')
sentiment_by_gender.plot(kind='bar', stacked=True)
plt.title('Sentiment Categories by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig("output/gender/sentiment_categories_by_gender.png")
plt.close()
#-------------------------------------------------------------------------------------------------


TTest_df = pd.read_csv('output/gender/gender_ttest_results.csv')
print(TTest_df.head())
# Use 't-statistic' and 'p-value' as features for clustering

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Create a new figure
plt.figure(figsize=(10, 8))
m = Basemap(projection='lcc', resolution='h', # Set up Basemap
            lat_0=37.5, lon_0=-96,
            llcrnrlon=-119, llcrnrlat=22,
            urcrnrlon=-64, urcrnrlat=49)

m.shadedrelief()  # Add a shaded relief map
m.drawcoastlines()
m.drawcountries()
# Example coordinates (longitude, latitude)
lon = [-100, -80, -90]
lat = [40, 30, 35]
m.scatter(lon, lat, latlon=True, marker='o', color='r')
plt.title('Map with Points')
plt.savefig('output/gender/map_plot.png')  # Save as PNG
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Scatter plot with regression line
sns.scatterplot(data=TTest_df, x='t-statistic', y='p-value', hue='genderGroup1')
plt.title('T-Statistic vs P-Value by Region')
plt.xlabel('T-Statistic')
plt.ylabel('P-Value')
plt.savefig('output/gender/seaborn_scatterplot.png')
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Set significance threshold for marking
significance_threshold = 0.05
# Create a figure with subplots to match the style in the picture
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
# 1. Bar Plot of t-statistics
sns.barplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df, ax=axs[0, 0])
axs[0, 0].set_title('T-Statistics Between Regions')
axs[0, 0].set_ylabel('T-Statistic')
# Annotating significance on the bar plot
for i in range(TTest_df.shape[0]):
    p_value = TTest_df['p-value'][i]
    if p_value < significance_threshold:
        axs[0, 0].text(i, TTest_df['t-statistic'][i] + 0.2, '*', ha='center', va='bottom', color='red', fontsize=14)
# 2. KDE Plot to show distribution for t-statistics
for region_pair in TTest_df['genderGroup1'].unique():
    subset = TTest_df[TTest_df['genderGroup1'] == region_pair]
if subset['t-statistic'].var() > 0:
    sns.kdeplot(subset['t-statistic'], fill=True, ax=axs[0, 1], label=region_pair)
else:
    print(f"Skipping KDE plot for {region_pair} due to zero variance in t-statistic")
axs[0, 1].set_title('Density Plot for T-Statistics')
axs[0, 1].legend(title='Region Pairs')
# 3. Box Plot for regions comparison
sns.boxplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df, ax=axs[1, 0])
axs[1, 0].set_title('T-Statistics Comparison Boxplot')
# 4. Distribution comparison (Similar to picture provided)
sns.violinplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df, split=True, ax=axs[1, 1])
axs[1, 1].set_title('Violin Plot of Region Comparisons')
# Adjust layout
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Set significance threshold for marking
significance_threshold = 0.05
# 1. Bar Plot of t-statistics
plt.figure(figsize=(8, 6))
sns.barplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df)
plt.title('T-Statistics Between Regions')
plt.ylabel('T-Statistic')
# Annotating significance on the bar plot
for i in range(TTest_df.shape[0]):
    p_value = TTest_df['p-value'][i]
    if p_value < significance_threshold:
        plt.text(i, TTest_df['t-statistic'][i] + 0.2, '*', ha='center', va='bottom', color='red', fontsize=14)
# Save bar plot
plt.savefig('output/gender/t_statistics_bar_plot.png', bbox_inches='tight')
plt.clf()  # Clear the current figure
# 2. KDE Plot to show distribution for t-statistics
plt.figure(figsize=(8, 6))
for region_pair in TTest_df['genderGroup1'].unique():
    subset = TTest_df[TTest_df['genderGroup1'] == region_pair]
    if subset['t-statistic'].var() > 0:
        sns.kdeplot(subset['t-statistic'], fill=True, label=region_pair)
    else:
        print(f"Skipping KDE plot for {region_pair} due to zero variance in t-statistic")
plt.title('Density Plot for T-Statistics')
plt.legend(title='Region Pairs')
# Save KDE plot
plt.savefig('output/gender/t_statistics_kde_plot.png', bbox_inches='tight')
plt.clf()
# 3. Box Plot for regions comparison
plt.figure(figsize=(8, 6))
sns.boxplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df)
plt.title('T-Statistics Comparison Boxplot')
# Save box plot
plt.savefig('output/gender/t_statistics_box_plot.png', bbox_inches='tight')
plt.clf()
# 4. Violin Plot for distribution comparison
plt.figure(figsize=(8, 6))
sns.violinplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df, split=True)
plt.title('Violin Plot of Region Comparisons')
# Save violin plot
plt.savefig('output/gender/t_statistics_violin_plot.png', bbox_inches='tight')
plt.clf()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Calculate the median t-statistic for each region pair
median_values = TTest_df.groupby(['genderGroup1', 'genderGroup2']).median()['t-statistic'].reset_index()
# Label regions as nodes
regions = list(set(TTest_df['genderGroup1']) | set(TTest_df['genderGroup2']))
# Map regions to indices (for the sankey diagram)
region_index = {region: idx for idx, region in enumerate(regions)}
# Create source and target lists based on region indices
source = [region_index[genderGroup1] for genderGroup1 in median_values['genderGroup1']]
target = [region_index[genderGroup2] for genderGroup2 in median_values['genderGroup2']]
values = median_values['t-statistic'].abs()  # Use absolute values for better visualization
# Create a color scale based on t-statistic values
color_scale = plt.cm.RdYlBu  # Red-Yellow-Blue color scale
normalized_values = (values - values.min()) / (values.max() - values.min())
link_colors = [color_scale(val) for val in normalized_values]
# Sankey Diagram construction
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=regions,
        color="lightblue"
    ),
    link=dict(
        source=source,
        target=target,
        value=values,
        color=[f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.5)" for r, g, b, _ in link_colors]
    )
))

fig.update_layout(
    title_text="Median Flow of T-Statistics Between Regions",
    font_size=12,
    width=800,
    height=600
)
fig.write_image('output/gender/Median_Flow_Chart.png', scale=2)  # Increased resolution
fig.show()
#--------------------------------------------------------------------------------
# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='t-statistic', y='genderGroup1', hue='genderGroup2', data=TTest_df, orient='h')
plt.title('T-Statistics Between Genders (Horizontal Bar Chart)', fontsize=16)
plt.xlabel('T-Statistic', fontsize=12)
plt.ylabel('Gender 1', fontsize=12)
# Add a legend
plt.legend(title='Gender 2', title_fontsize='12', fontsize='10', loc='center left', bbox_to_anchor=(1, 0.5))
# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.savefig('output/gender/horizontal_bar_chart.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Pivot the data to create a matrix
pivot_df = TTest_df.pivot(index='genderGroup1', columns='genderGroup2', values='t-statistic')
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(pivot_df, dtype=bool))
# Set up the matplotlib figure
plt.figure(figsize=(12, 10))
# Create the heatmap
sns.heatmap(pivot_df, mask=mask, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Pairwise T-Test Results Between Genders', fontsize=16)
plt.tight_layout()
# Save the plot
plt.savefig('output/gender/pairwise_ttest_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
# Create a second heatmap for p-values
pivot_df_p = TTest_df.pivot(index='genderGroup1', columns='genderGroup2', values='p-value')
plt.figure(figsize=(12, 10))
# Use a different colormap for p-values
sns.heatmap(pivot_df_p, mask=mask, annot=True, cmap='YlOrRd_r', 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Pairwise T-Test P-Values Between Genders', fontsize=16)
plt.tight_layout()
# Save the plot
plt.savefig('output/gender/pairwise_ttest_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Create a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df)
plt.title('Distribution of T-Statistics Between Genders', fontsize=16)
plt.xlabel('Gender 1', fontsize=12)
plt.ylabel('T-Statistic', fontsize=12)
# Rotate x-axis labels if needed
plt.xticks(rotation=45, ha='right')
# Add a legend
plt.legend(title='Gender 2', title_fontsize='12', fontsize='10', loc='best')
# Adjust layout to prevent cutting off labels
plt.tight_layout()
# Save the plot
plt.savefig('output/gender/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Group the data by genderGroup1 and genderGroup2, and calculate the mean t-statistic
grouped_data = TTest_df.groupby(['genderGroup1', 'genderGroup2'])['t-statistic'].mean().abs().reset_index()
# Create two separate DataFrames for genderGroup1 and genderGroup2
genderGroup1_data = grouped_data.groupby('genderGroup1')['t-statistic'].sum().reset_index()
genderGroup2_data = grouped_data.groupby('genderGroup2')['t-statistic'].sum().reset_index()
# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
# Plot for Gender 1
ax1.pie(genderGroup1_data['t-statistic'], labels=genderGroup1_data['genderGroup1'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Distribution of T-Statistics by Gender 1')
# Plot for Gender 2
ax2.pie(genderGroup2_data['t-statistic'], labels=genderGroup2_data['genderGroup2'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribution of T-Statistics by Gender 2')
# Adjust layout and save
plt.tight_layout()
plt.savefig('output/gender/gender_pie_charts.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------       
# Create a box plot
plt.figure(figsize=(14, 10))
sns.boxplot(x='genderGroup1', y='t-statistic', hue='genderGroup2', data=TTest_df, 
            palette='Set3', width=0.8, fliersize=5)
plt.title('Distribution of T-Statistics Between Regions', fontsize=18)
plt.xlabel('Region 1', fontsize=14)
plt.ylabel('T-Statistic', fontsize=14)
# Rotate x-axis labels if needed
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
# Add a legend
plt.legend(title='Region 2', title_fontsize='14', fontsize='12', loc='best')
# Add a text explanation
explanation = """
Box Plot Elements:
- Box: Interquartile Range (IQR)
- Line in Box: Median
- Whiskers: Extend to 1.5 * IQR
- Points: Outliers beyond whiskers
"""
plt.text(0.02, -0.15, explanation, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the plot
plt.savefig('output/gender/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
fig, ax = plt.subplots(figsize=(12, 6))
# Create the box plot
bp = ax.boxplot(TTest_df.groupby('genderGroup1')['t-statistic'].apply(list), 
                patch_artist=True, 
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='black', markersize=6,
                                linestyle='none'))

# Customize the plot
ax.set_title('Distribution of T-Statistics Across Regions', fontsize=16)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('T-Statistic', fontsize=12)
# Set x-tick labels
ax.set_xticklabels(TTest_df['genderGroup1'].unique(), rotation=45, ha='right')
# Add a text explanation
explanation = """
Box Plot Elements:
- Box: Interquartile Range (IQR)
- Red Line: Median
- Whiskers: Extend to min/max within 1.5 * IQR
- Black Dots: Outliers beyond whiskers
"""
plt.text(0.02, -0.2, explanation, transform=ax.transAxes, 
         fontsize=10, verticalalignment='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig('output/gender/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Group the data by genderGroup1 and calculate the mean t-statistic
grouped_data = TTest_df.groupby('genderGroup1')['t-statistic'].mean().sort_values(ascending=False)
# Create the line graph
plt.figure(figsize=(12, 6))
plt.plot(grouped_data.index, grouped_data.values, marker='o')
plt.title('Average T-Statistics Across Regions', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average T-Statistic', fontsize=12)
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Add value labels on the points
for i, v in enumerate(grouped_data.values):
    plt.text(i, v, f'{v:.2f}', ha='left', va='bottom')
# Adjust layout and save
plt.tight_layout()
plt.savefig('output/gender/t_statistics_line_graph.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
pivot_df = TTest_df.pivot(index='genderGroup1', columns='genderGroup2', values='t-statistic')
# Create the line graph
plt.figure(figsize=(14, 8))
sns.lineplot(data=pivot_df, markers=True, dashes=False)
plt.title('T-Statistics Across Regions', fontsize=16)
plt.xlabel('Region 1', fontsize=12)
plt.ylabel('T-Statistic', fontsize=12)
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Move legend outside the plot
plt.legend(title='Region 2', bbox_to_anchor=(1.05, 1), loc='upper left')
# Adjust layout and save
plt.tight_layout()
plt.savefig('output/gender/t_statistics_multi_line_graph.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
plt.figure(figsize=(10, 10))
# Use t-statistic as x and p-value as y (you may need to adjust based on your actual data)
sns.scatterplot(data=TTest_df, x='t-statistic', y='p-value', hue='genderGroup1', 
                palette='Set1', s=50, alpha=0.7)
plt.title('T-Statistic vs P-Value by Region', fontsize=16)
plt.xlabel('T-Statistic', fontsize=12)
plt.ylabel('P-Value', fontsize=12)
# Set log scale for y-axis (common in flow cytometry plots)
plt.yscale('log')
# Add quadrant lines (adjust these values based on your data)
plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
# Adjust legend
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
# Adjust layout and save
plt.tight_layout()
plt.savefig('output/gender/flow_cytometry_style_plot.png', bbox_inches='tight', dpi=300)
plt.show()

#--------------------------------------------------------------------------------
#box plot comparing 3 groups 
# matplotlib library, seaborn
# Calculate the error
tukey_data = pd.read_csv('output/gender/gender_tukey_results.csv')
tukey_data['yerr'] = (tukey_data['upper'] - tukey_data['lower']) / 2

# Set up the plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create the box plot
ax = sns.boxplot(x='group1', y='meandiff', data=tukey_data, color='lightblue')

# Add error bars
ax.errorbar(x=range(len(tukey_data)), y=tukey_data['meandiff'], yerr=tukey_data['yerr'], fmt='none', c='black', capsize=5)

# Customize the plot
plt.title('Mean Differences Between Groups with Error Bars')
plt.xlabel('Group 1')
plt.ylabel('Mean Difference')

# Add text annotations for p-value and reject status
for i, row in tukey_data.iterrows():
    plt.text(i, row['upper'], f"p={row['p-adj']:.3f}\nreject={row['reject']}", 
             ha='center', va='bottom')

# Add line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='--')

# Show the plot
plt.tight_layout()
plt.savefig('output/gender/Tukey_custom_box_plot.png', dpi=300, bbox_inches='tight')
plt.show()
#--------------------------------------------------------------------------------
# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Expression (RPRT) across cell stages', fontsize=16)

# X-axis labels
x_labels = ['4cell', '8cell', '16cell', '32cell', '64cell']

# Generate some mock data to mimic the image
np.random.seed(42)
female_wt = np.random.normal(15, 5, 5)
female_ko = np.random.normal(20, 7, 5)
male_wt = np.random.normal(25, 10, 5)

# Plot for Female: WT
ax1.plot(x_labels, female_wt, 'o-', color='blue', label='WT')
ax1.fill_between(x_labels, female_wt - 2, female_wt + 2, alpha=0.2, color='blue')
ax1.set_title('Female: WT')
ax1.set_ylim(0, 50)
ax1.set_ylabel('Expression (RPRT)')

# Plot for Female: Xist KO
ax2.plot(x_labels, female_ko, 'o-', color='green', label='Xist KO')
ax2.fill_between(x_labels, female_ko - 2, female_ko + 2, alpha=0.2, color='green')
ax2.set_title('Female: Xist KO')
ax2.set_ylim(0, 50)

# Plot for Male: WT
ax3.plot(x_labels, male_wt, 'o-', color='blue', label='WT')
ax3.fill_between(x_labels, male_wt - 2, male_wt + 2, alpha=0.2, color='blue')
ax3.set_title('Male: WT')
ax3.set_ylim(0, 50)

# Customize the plots
for ax in (ax1, ax2, ax3):
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('output/gender/expression_across_cell_stages.png', dpi=300, bbox_inches='tight')
plt.show()