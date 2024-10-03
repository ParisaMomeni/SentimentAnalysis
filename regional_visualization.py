import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('output/regional/Regional_ttest_results.csv')
print(df.head())
# Use 't-statistic' and 'p-value' as features for clustering
X = df[['t-statistic', 'p-value']]
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.savefig('output/regional/Elbow.png')  # Change the path and filename as needed
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Create a new figure
# Define region coordinates (approximate centers)
region_coords = {
    'West': (-120, 40),
    'Northeast': (-74, 42),
    'Midwest': (-90, 42),
    'Southeast': (-84, 33),
    'Southwest': (-105, 34),
    'Territory': (-66, 18)  # Assuming this is for Puerto Rico
}

# Create the map
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-130, -60, 20, 50], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Plot connections between regions
for _, row in df.iterrows():
    start = region_coords[row['region1']]
    end = region_coords[row['region2']]
    plt.plot([start[0], end[0]], [start[1], end[1]], 
             color='red' if row['p-value'] < 0.05 else 'blue',
             linewidth=2 * abs(row['t-statistic']),
             transform=ccrs.PlateCarree(),
             alpha=0.6)

# Plot region points
for region, coords in region_coords.items():
    ax.plot(coords[0], coords[1], 'ko', markersize=10, transform=ccrs.PlateCarree())
    ax.text(coords[0], coords[1], region, fontsize=10, ha='center', va='bottom', 
            transform=ccrs.PlateCarree(), bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.title('Regional T-Test Results Visualization', fontsize=16)

# Add a legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=2, label='p < 0.05'),
                   Line2D([0], [0], color='blue', lw=2, label='p >= 0.05')]
ax.legend(handles=legend_elements, loc='lower left')

plt.savefig('output/regional/regional_ttest_map.png', dpi=300, bbox_inches='tight')
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Scatter plot with regression line
sns.scatterplot(data=df, x='t-statistic', y='p-value', hue='region1')
plt.title('T-Statistic vs P-Value by Region')
plt.xlabel('T-Statistic')
plt.ylabel('P-Value')
plt.savefig('output/regional/seaborn_scatterplot.png')
plt.show()

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(data=df, x='t-statistic', y='p-value', hue='region1', style='region2', s=100)
plt.title('T-Statistic vs P-Value by Region Comparison', fontsize=16)
plt.xlabel('T-Statistic', fontsize=14)
plt.ylabel('P-Value', fontsize=14)

# Add significance line
plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')

# Annotate points
for idx, row in df.iterrows():
    plt.annotate(f"{row['region1']}-{row['region2']}", 
                 (row['t-statistic'], row['p-value']),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.legend(title='Region 1', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/regional/enhanced_scatterplot.png', dpi=300, bbox_inches='tight')
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Set significance threshold for marking
significance_threshold = 0.05
# Create a figure with subplots to match the style in the picture
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
# 1. Bar Plot of t-statistics
sns.barplot(x='region1', y='t-statistic', hue='region2', data=df, ax=axs[0, 0])
axs[0, 0].set_title('T-Statistics Between Regions')
axs[0, 0].set_ylabel('T-Statistic')
# Annotating significance on the bar plot
for i in range(df.shape[0]):
    p_value = df['p-value'][i]
    if p_value < significance_threshold:
        axs[0, 0].text(i, df['t-statistic'][i] + 0.2, '*', ha='center', va='bottom', color='red', fontsize=14)
# 2. KDE Plot to show distribution for t-statistics
for region_pair in df['region1'].unique():
    subset = df[df['region1'] == region_pair]
    sns.kdeplot(subset['t-statistic'], fill=True, ax=axs[0, 1], label=region_pair)
axs[0, 1].set_title('Density Plot for T-Statistics')
axs[0, 1].legend(title='Region Pairs')
# 3. Box Plot for regions comparison
sns.boxplot(x='region1', y='t-statistic', hue='region2', data=df, ax=axs[1, 0])
axs[1, 0].set_title('T-Statistics Comparison Boxplot')
# 4. Distribution comparison (Similar to picture provided)
sns.violinplot(x='region1', y='t-statistic', hue='region2', data=df, split=True, ax=axs[1, 1])
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
sns.barplot(x='region1', y='t-statistic', hue='region2', data=df)
plt.title('T-Statistics Between Regions')
plt.ylabel('T-Statistic')
# Annotating significance on the bar plot
for i in range(df.shape[0]):
    p_value = df['p-value'][i]
    if p_value < significance_threshold:
        plt.text(i, df['t-statistic'][i] + 0.2, '*', ha='center', va='bottom', color='red', fontsize=14)
# Save bar plot
plt.savefig('output/regional/t_statistics_bar_plot.png', bbox_inches='tight')
plt.clf()  # Clear the current figure
# 2. KDE Plot to show distribution for t-statistics
plt.figure(figsize=(8, 6))
for region_pair in df['region1'].unique():
    subset = df[df['region1'] == region_pair]
    sns.kdeplot(subset['t-statistic'], fill=True, label=region_pair)
plt.title('Density Plot for T-Statistics')
plt.legend(title='Region Pairs')
# Save KDE plot
plt.savefig('output/regional/t_statistics_kde_plot.png', bbox_inches='tight')
plt.clf()
# 3. Box Plot for regions comparison
plt.figure(figsize=(8, 6))
sns.boxplot(x='region1', y='t-statistic', hue='region2', data=df)
plt.title('T-Statistics Comparison Boxplot')
# Save box plot
plt.savefig('output/regional/t_statistics_box_plot.png', bbox_inches='tight')
plt.clf()
# 4. Violin Plot for distribution comparison
plt.figure(figsize=(8, 6))
sns.violinplot(x='region1', y='t-statistic', hue='region2', data=df, split=True)
plt.title('Violin Plot of Region Comparisons')
# Save violin plot
plt.savefig('output/regional/t_statistics_violin_plot.png', bbox_inches='tight')
plt.clf()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Calculate the median t-statistic for each region pair
median_values = df.groupby(['region1', 'region2']).median()['t-statistic'].reset_index()
# Label regions as nodes
regions = list(set(df['region1']) | set(df['region2']))
# Map regions to indices (for the sankey diagram)
region_index = {region: idx for idx, region in enumerate(regions)}
# Create source and target lists based on region indices
source = [region_index[region1] for region1 in median_values['region1']]
target = [region_index[region2] for region2 in median_values['region2']]
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
fig.write_image('output/regional/Median_Flow_Chart.png', scale=2)  # Increased resolution
fig.show()
#--------------------------------------------------------------------------------
# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='t-statistic', y='region1', hue='region2', data=df, orient='h')
plt.title('T-Statistics Between Regions (Horizontal Bar Chart)', fontsize=16)
plt.xlabel('T-Statistic', fontsize=12)
plt.ylabel('Region 1', fontsize=12)
# Add a legend
plt.legend(title='Region 2', title_fontsize='12', fontsize='10', loc='center left', bbox_to_anchor=(1, 0.5))
# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.savefig('output/regional/horizontal_bar_chart.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Pivot the data to create a matrix
pivot_df = df.pivot(index='region1', columns='region2', values='t-statistic')
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(pivot_df, dtype=bool))
# Set up the matplotlib figure
plt.figure(figsize=(12, 10))
# Create the heatmap
sns.heatmap(pivot_df, mask=mask, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Pairwise T-Test Results Between Regions', fontsize=16)
plt.tight_layout()
# Save the plot
plt.savefig('output/regional/pairwise_ttest_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
# Create a second heatmap for p-values
pivot_df_p = df.pivot(index='region1', columns='region2', values='p-value')
plt.figure(figsize=(12, 10))
# Use a different colormap for p-values
sns.heatmap(pivot_df_p, mask=mask, annot=True, cmap='YlOrRd_r', 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Pairwise T-Test P-Values Between Regions', fontsize=16)
plt.tight_layout()
# Save the plot
plt.savefig('output/regional/pairwise_ttest_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Create a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='region1', y='t-statistic', hue='region2', data=df)
plt.title('Distribution of T-Statistics Between Regions', fontsize=16)
plt.xlabel('Region 1', fontsize=12)
plt.ylabel('T-Statistic', fontsize=12)
# Rotate x-axis labels if needed
plt.xticks(rotation=45, ha='right')
# Add a legend
plt.legend(title='Region 2', title_fontsize='12', fontsize='10', loc='best')
# Adjust layout to prevent cutting off labels
plt.tight_layout()
# Save the plot
plt.savefig('output/regional/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Group the data by region1 and region2, and calculate the mean t-statistic
grouped_data = df.groupby(['region1', 'region2'])['t-statistic'].mean().abs().reset_index()
# Create two separate DataFrames for region1 and region2
region1_data = grouped_data.groupby('region1')['t-statistic'].sum().reset_index()
region2_data = grouped_data.groupby('region2')['t-statistic'].sum().reset_index()
# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
# Plot for Region 1
ax1.pie(region1_data['t-statistic'], labels=region1_data['region1'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Distribution of T-Statistics by Region 1')
# Plot for Region 2
ax2.pie(region2_data['t-statistic'], labels=region2_data['region2'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribution of T-Statistics by Region 2')
# Adjust layout and save
plt.tight_layout()
plt.savefig('output/regional/regional_pie_charts.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------       
# Create a box plot
plt.figure(figsize=(14, 10))
sns.boxplot(x='region1', y='t-statistic', hue='region2', data=df, 
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
plt.savefig('output/regional/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------  
# to do use Regional_region_group_mean.csv 
fig, ax = plt.subplots(figsize=(12, 6))
# Create the box plot
bp = ax.boxplot(df.groupby('region1')['t-statistic'].apply(list), 
                patch_artist=True, 
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='black', markersize=6,
                                linestyle='none'))

# Customize the plot
ax.set_title('Distribution of T-Statistics Across Regions', fontsize=16)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('T-Statistic', fontsize=12)
# Set x-tick labels
ax.set_xticklabels(df['region1'].unique(), rotation=45, ha='right')
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
plt.savefig('output/regional/t_statistics_box_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
# Group the data by region1 and calculate the mean t-statistic
grouped_data = df.groupby('region1')['t-statistic'].mean().sort_values(ascending=False)
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
plt.savefig('output/regional/t_statistics_line_graph.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------   
pivot_df = df.pivot(index='region1', columns='region2', values='t-statistic')
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
plt.savefig('output/regional/t_statistics_multi_line_graph.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------
plt.figure(figsize=(10, 10))
# Use t-statistic as x and p-value as y (you may need to adjust based on your actual data)
sns.scatterplot(data=df, x='t-statistic', y='p-value', hue='region1', 
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
plt.savefig('output/regional/flow_cytometry_style_plot.png', bbox_inches='tight', dpi=300)
plt.show()
#--------------------------------------------------------------------------------