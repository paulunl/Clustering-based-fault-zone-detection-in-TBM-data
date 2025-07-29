# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:11:50 2025

@author: Alienhirn
"""

# =============================================================================
# Clustering-based anomaly detection
# =============================================================================

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

df = pd.read_parquet(r'02_processed_data\02_TBMdata_BBT_S_preprocessed_wlabels.gzip')

# filter certain sections based on Tunnel Distance
FROM = 13900
TO = 26500
outlier_fraction = 0.005

df = df[(df['Tunnel Distance [m]'] > FROM) & (df['Tunnel Distance [m]'] < TO)]

# check correlations with pairplot
# sns.pairplot(df)

# choose features to cluster
columns_to_cluster = ['Specific Energy [MJ/m³]',
                      'Spec. Penetration [mm/rot/MN]',
                      'torque ratio']

df_clust = df[columns_to_cluster]

# Scale the data using standardscaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clust)

n_clusters = range(1, 11)

# calculate inertia for different cluster numbers
inertia = []
for i in n_clusters:
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(df_scaled)
    inertia.append(model.inertia_)

# Find the optimal number of clusters using KneeLocator
knee_locator = KneeLocator(n_clusters,
                           inertia,
                           curve="convex",
                           direction="decreasing")
optimal_clusters = knee_locator.knee

# Plot the elbow curve
plt.plot(n_clusters, inertia, marker='o')
plt.axvline(x=optimal_clusters, color='r', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.legend()

print(f"The optimal number of clusters is: {optimal_clusters}")

# Determine Fault Zones
# Fit the KMeans model
kmeans = KMeans(n_clusters=optimal_clusters, random_state=69)
kmeans.fit(df_scaled)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Calculate distance from each point to its nearest centroid
_, distances = pairwise_distances_argmin_min(df_scaled, centroids)

# Determine Threshold for Fault Zones
# Set outliers_fraction
# (e.g., 0.1 indicates 10% of points are considered outliers)
outliers_fraction = outlier_fraction
number_of_outliers = int(len(distances) * outliers_fraction)

# Find the threshold: minimum distance of the top "outliers_fraction" most distant points
threshold = np.partition(distances, -number_of_outliers)[-number_of_outliers]

# Mark Fault Zones
# Mark points as fault zones (1) or normal (0) based on the threshold
fault_zones = np.where(distances >= threshold, 1, 0)

# Create a new column in the original DataFrame for fault zones
df['Fault Zone Cluster'] = fault_zones
df['cluster_prob_score'] = distances

# Plot results
def get_cmap(n_classes):
    """Returns a color map based on the number of classes."""
    color_sets = {
        6: ['white', 'green', 'greenyellow', 'gold', 'orange', 'red'],
        5: ['white', 'green', 'greenyellow', 'gold', 'orange'],
        4: ['white', 'green', 'greenyellow', 'gold'],
        3: ['white', 'green', 'greenyellow'],
        2: ['white', 'green'],
        1: ['red']
    }
    if n_classes in color_sets:
        return mpl.colors.ListedColormap(color_sets[n_classes])
    else:
        raise ValueError(f"No suitable color map for {n_classes} n_classes")

def add_class_overlay(ax, df, cmap, n_classes, xlim):
    """Add class-colored overlay and colorbar to a plot axis."""
    c = ax.pcolorfast(xlim, ax.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=df['Class'].min(),
                      vmax=df['Class'].max() + 1,
                      alpha=0.5)
    cbar = plt.colorbar(c, ax=ax)
    
    ticks = np.linspace(df['Class'].min(), df['Class'].max() + 1, 2 * n_classes + 1)[1::2]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(range(int(df['Class'].min()), int(df['Class'].max()) + 1))
    cbar.ax.set_ylim(df['Class'].min(), df['Class'].max() + 1)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Class", fontsize=12)

def plot_metric(ax, df, fault_zones, x, y, ylabel, ylim, window, xlim):
    """Helper to plot metric with rolling mean, raw data, and anomaly markers."""
    ax.plot(df[x], df[y].rolling(window).mean(), color='black',
            label='Rolling Mean', zorder=1)
    ax.plot(df[x], df[y], color='black', alpha=0.5, label='Normal')
    ax.scatter(fault_zones[x], fault_zones[y], color='red',
               label='Clustered Fault Zone',
               alpha=0.8, edgecolor='black')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    
def visualize_clustering_anomalies(df, file_name, rolling_window=50):
    """Visualizes clustering-based anomalies in a DataFrame."""

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8))
    fault_zones = df[df['Fault Zone Cluster'] == 1]
    x = 'Tunnel Distance [m]'
    xlim = (df[x].min(), df[x].max())
    xticks = np.arange(xlim[0], xlim[1] + 1, 50)  # +1 to include endpoint
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(xticks)
        
    # Plot 1: Specific Energy
    plot_metric(ax1, df, fault_zones, x, 'Specific Energy [MJ/m³]',
                ylabel='Specific Energy \n[MJ/m³]', ylim=(0, 250),
                window=rolling_window, xlim=xlim)

    # Plot 2: Specific Penetration
    plot_metric(ax2, df, fault_zones, x, 'Spec. Penetration [mm/rot/MN]',
                ylabel='Spec. Penetration \n[mm/rot/MN]', ylim=(0, 10),
                window=rolling_window, xlim=xlim)

    # Plot 3: Torque Ratio
    plot_metric(ax3, df, fault_zones, x, 'torque ratio',
                ylabel='Torque Ratio', ylim=(0, 3),
                window=rolling_window, xlim=xlim)
    ax3.set_xlabel(x, fontsize=12)

    # Cluster color overlay
    n_clusters = int(df['Class'].max() - df['Class'].min() + 1)
    cmap = get_cmap(n_clusters)

    for ax in [ax1, ax2, ax3]:
        add_class_overlay(ax, df, cmap, n_clusters, xlim)
    
    # Identify fault zone transitions
    fault_diff = df['Fault'].fillna(0).astype(int).diff()
    
    # Indices where fault starts (0 → 1)
    fault_starts = df.loc[fault_diff == 1, 'Tunnel Distance [m]']
    
    # Indices where fault ends (1 → 0)
    fault_ends = df.loc[fault_diff == -1, 'Tunnel Distance [m]']
    
    # Plot vertical lines for mapped fault zones
    for ax in [ax1, ax2, ax3]:
        label_added = False
        red_zone_pairs = list(zip(fault_starts, fault_ends))
        
        for i, (start, end) in enumerate(red_zone_pairs):
            ax.axvline(start, color='blue', linestyle='--', linewidth=1.5,
                       label='Mapped Fault Zone' if not label_added else None)
            ax.axvline(end, color='blue', linestyle='--', linewidth=1.5)
            ax.axvspan(start, end, facecolor='none', edgecolor='blue',
                       hatch='//', linewidth=0, alpha=0.4)
            label_added = True

        '''   
        # plot vertical lines for clustering detected fault zones >= 25 consecutive rows
        
        in_block = False
        count = 0
        start_idx = None
        label_added2 = False
        threshold = 10
        
        for i in range(len(df)):

            if df.iloc[i]['Fault Zone Cluster'] == 1:
                if not in_block:
                    in_block = True
                    start_idx = i
                count += 1
            else:
                if in_block and count >= threshold:
                    start_x = df.iloc[start_idx][x]
                    end_x = df.iloc[i - 1][x]
                    ax.axvline(start_x, color='blue', linestyle='--', linewidth=1.5,
                               label='Clustered Fault Zone' if not label_added2 else None)
                    label_added2 = True
                    ax.axvline(end_x, color='blue', linestyle='--', linewidth=1.5)
                # Reset state
                in_block = False
                count = 0
    
        # Check for case when the block goes till the end of the DataFrame
        if in_block and count >= threshold:
            start_x = df.iloc[start_idx][x]
            end_x = df.iloc[len(df) - 1][x]
            ax.axvline(start_x, color='blue', linestyle='--', linewidth=1.5,
                       label='Clustered Fault Zone' if not label_added2 else None)
            ax.axvline(end_x, color='blue', linestyle='--', linewidth=1.5)
        '''
    ax1.legend(loc='center left', bbox_to_anchor=(1.15, -0.7), fontsize=12)

    plt.savefig(f'03_Plots/01_clustering/{file_name}_.png', dpi=300,
                bbox_inches='tight')

visualize_clustering_anomalies(df, f'{FROM}_{TO}')
