import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sd_tn_thresholds(fulldata, dtype, sensor_labels,  save_name, title, ylabel=None,
                     highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212], height=10, width=15, font_size=16, palette='husl'):
    """
    Plot the summer days data with thresholds on the x-axis and color highlighting for selected stations.
    :param height:
    :param width:
    :param data: The summer days dataset, where 'threshold' is along the x-axis.
    :param title: The title of the plot.
    :param save_name: The name of the file to save the plot.
    :param highlight_stations: List of stations to highlight in color.
    """
    plt.rcParams.update({'font.size': font_size})
    # Extract the mean along the 'time' dimension to get summer days count per station for each threshold

    # Generate color palette for the highlighted stations
    palette = sns.color_palette(palette, len(highlight_stations))
    sns.set_palette(palette)

    data = fulldata[dtype].sum(dim='time')

    # Plot settings
    plt.figure(figsize=(width, height))
    remove_sensors = [240, 206, 207]


    # Plot all stations in gray
    for station in data.sensor:
        if station not in highlight_stations:
            if station not in remove_sensors:
                data.sel(sensor=station).plot.line(x='threshold', color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for idx, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        data.sel(sensor=station).plot.line(x='threshold', label=label, linewidth=2)

    # Customizing the plot
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Threshold (°C)', fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    # Beautify the plot
    plt.tight_layout()

    plt.savefig(save_name)

    # Show plot
    plt.close()


def plot_sd_tn_cumulative_sums(fulldata, dtype, threshold, sensor_labels, save_name, title,
                               highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212], height=10, width=15,
                               font_size=16, palette='husl'):
    plt.rcParams.update({'font.size': font_size})
    data = fulldata.sel(threshold=threshold).cumsum(dim='time')[dtype]
    plt.figure(figsize=(width, height))
    remove_sensors = [240, 206, 207]

    # Check how many times each station is being plotted
    plotted_stations = []

    # Plot all stations in gray
    for station in data.sensor:
        if station not in highlight_stations and station not in remove_sensors:
            plotted_stations.append(station.item())
            data.sel(sensor=station).plot.line(x='time', color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for station in highlight_stations:
        name = sensor_labels.get(station, f'Sensor {station}')
        label = f'Sensor {station} \n{name}'
        plotted_stations.append(station)
        data.sel(sensor=station).plot.line(x='time', label=label, linewidth=2)

    # Customizing the plot
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Time', fontsize=font_size)
    plt.ylabel(f'Total {dtype}', fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    # Beautify the plot
    plt.tight_layout()

    plt.savefig(save_name)

    # Show plot
    plt.close()

def plot_sd_excess_sums(fulldata, threshold, sensor_labels, save_name, title, ylabel=None, highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212], height=10, width=15, font_size=16, palette='husl'):
    plt.rcParams.update({'font.size': font_size})
    mydata = fulldata['summer_days'].sel(threshold=threshold).astype(int).cumsum(dim='time')
    data = mydata - mydata.sel(sensor=206)
    plt.figure(figsize=(width, height))
    remove_sensors = [240, 206, 207]

    # Plot all stations in gray
    for station in data.sensor:
        if station not in highlight_stations:
            if station not in remove_sensors:
                data.sel(sensor=station).plot.line(x='time', color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for idx, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        data.sel(sensor=station).plot.line(x='time', label=label, linewidth=2)

    # Customizing the plot
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Time', fontsize=font_size)
    plt.ylabel(f'Excess Summer Days', fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    # Beautify the plot
    plt.tight_layout()

    plt.savefig(save_name)

    # Show plot
    plt.close()