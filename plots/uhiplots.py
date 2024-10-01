from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy


def uhi_lineplot(data, highlight_stations, period, directories, sensor_labels, title):
    uhi_hourly_mean = data.groupby(data.time.dt.hour).mean()
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    palette = sns.color_palette("husl", len(highlight_stations))
    sns.set_palette(palette)
    # Plot settings
    plt.figure(figsize=(15, 9))

    # Plot all stations in gray
    for station in uhi_hourly_mean.sensor:
        if station not in highlight_stations:
            uhi_hourly_mean['uhi_206'].sel(sensor=station).plot.line(color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for station in highlight_stations:
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        uhi_hourly_mean['uhi_206'].sel(sensor=station).plot.line(label=label, linewidth=2)

    # Customizing the plot
    plt.title(title)
    plt.xlabel('Hour of the Day')
    plt.ylabel('UHI (°C)')
    plt.grid(True)

    # Set legend outside of the plot - maybe change
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Hour of the Day', fontsize=font_size)
    plt.ylabel('UHI Value', fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    directory = directories['plot_directory'] / 'uhiplots'
    directory.mkdir(parents=True, exist_ok=True)
    save_name = directory / f'uhi_lineplot_{period}.png'
    plt.savefig(save_name)
    plt.close()

    directory = directories['data_output'] / 'uhi'
    directory.mkdir(parents=True, exist_ok=True)
    uhi_hourly_mean.to_dataframe().reset_index().to_csv(directory / f'uhi_hourly_{period}.csv', index=False)

def prepare_for_qgis(data, sensor_names, directories, period):
    hourly_uhim = data.groupby(data.time.dt.hour).mean()
    hdfm = hourly_uhim.to_dataframe().reset_index()
    joined = hdfm.merge(sensor_names, on='sensor')
    directory = directories['data_output'] / 'uhi' / 'hourly'
    directory.mkdir(parents=True, exist_ok=True)
    for hour in range(0,24):
        joined[joined.hour == hour].to_csv(directory / 'uhi_{period}_{hour}.csv', index=False)


def plot_up_close(data, sensor_labels, title, highlight_stations, period, directories):
    cut_data = data.resample(time='1h').mean()
    plt.figure(figsize=(15, 9))
    for station in cut_data.sensor:
        if station not in highlight_stations:
            cut_data.sel(sensor=station).plot.line(color='gray', alpha=0.3, label='_nolegend_')
    for station in highlight_stations:
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        cut_data.sel(sensor=station).plot.line(label=label, linewidth=2)
    plt.title(title, fontsize=18)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('UHI (°C)', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.tight_layout()
    directory = directories['plot_directory'] / 'uhiplots'
    directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(directory / f"uhi_lineplot_{period}_{'_'.join([str(x) for x in highlight_stations])}.png")
    plt.close()


def plot_daily_max_min(data, sensor_labels, title, highlight_stations, period, directories):
    dmax = data.sel(sensor = [x for x in data.sensor if x not in [206, 207, 240]]).resample(time='1d').max()
    dmin = data.sel(sensor = [x for x in data.sensor if x not in [206, 207, 240]]).resample(time='1d').min()
    # Create subplots: one for max, one for min
    colormap = sns.color_palette("husl", len(highlight_stations))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))

    # Plot daily max
    for station in dmax.sensor:
        if station not in highlight_stations:
            dmax.sel(sensor=station).plot.line(ax=ax1, color='gray', alpha=0.3, label='_nolegend_')
    for index, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        dmax.sel(sensor=station).plot.line(ax=ax1, color=colormap[index], linewidth=2)
    ax1.set_title('Daily Maximum')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('UHI (°C)')

    # Plot daily min
    for station in dmin.sensor:
        if station not in highlight_stations:
            dmin.sel(sensor=station).plot.line(ax=ax2, color='gray', alpha=0.3, label='_nolegend_')
    print(highlight_stations)
    for index, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        dmin.sel(sensor=station).plot.line(ax=ax2, label=label, color=colormap[index], linewidth=2)
        print(label)
    ax2.set_title('Daily Minimum')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('UHI (°C)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Set overall title
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(directories['plot_directory'] / 'uhiplots' / f"uhi_daily_max_min_{period}.png")
    plt.close()