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
df = pd.read_csv('output/verified_users/verified_user_stats.csv')

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set the width of each bar and positions of the bars
width = 0.35
x = np.arange(len(df['User Type']))

# Create the bars
mean_bars = ax.bar(x - width/2, df['Mean'], width, label='Mean', color='skyblue')
median_bars = ax.bar(x + width/2, df['Median'], width, label='Median', color='lightgreen')

# Add error bars for standard deviation
ax.errorbar(x - width/2, df['Mean'], yerr=df['Std Dev'], fmt='none', color='blue', capsize=5)

# Customize the plot
ax.set_ylabel('Sentiment Score')
ax.set_title('Sentiment Analysis: Verified vs Not Verified Users')
ax.set_xticks(x)
ax.set_xticklabels(df['User Type'])
ax.legend()

# Add value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

add_value_labels(mean_bars)
add_value_labels(median_bars)

# Add sample size information
for i, count in enumerate(df['Count']):
    ax.text(i, ax.get_ylim()[0], f'n={count}', ha='center', va='top')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('output/verified_users/verified_user_statsplt.png')
plt.show()