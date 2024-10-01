from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy


def load_data():
    """
    Load the main and pilot data and filter out the 'ref' sensor.

    :return:
    """
    with xr.open_dataset(Path('/data/rawbieldata/main_data.nc')) as ds:
        mdata = ds.load()
    mdata = mdata.rename({'logger': 'sensor'}).temperature
    with xr.open_dataset(Path('/data/rawbieldata/pilot_data.nc')) as ds:
        pdata = ds.load()
    sensors = pdata.sensor.values
    sensors = [s for s in sensors if s != 'ref']
    pdata = pdata.sel(sensor=sensors)
    sensors = [int(s) for s in pdata.sensor]
    pdata['sensor'] = sensors
    # Filter out 'ref' coordinate and its data
    pdata = pdata.temp
    pdata.name = 'temperature'
    return mdata, pdata


def load_meteo():
    df = pd.read_table('/data/meteo/all_21_23.txt', sep=r'\s+', skiprows=1, dtype=str)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', errors='coerce')
    df = df[df.time.dt.year > 2021].copy()
    df['time'] = df['time'] + pd.Timedelta(hours=2)
    rename_cols = {
        'tre200s0': 'temperature',
        'rre150z0': 'precipitation',
        'ure200s0': 'humidity',
        'fve010z0': 'windspeed',
        'dkl010z0': 'winddir'
    }
    df = df.rename(columns=rename_cols)
    df = df.set_index(['time', 'stn'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ds = df.to_xarray()
    summer22 = ds.sel(time=slice('2022-07-09', '2022-09-10'))
    summer23 = ds.sel(time=slice('2023-05-15', '2023-09-16'))
    return summer22, summer23


sensor_labels = {
    206: 'rural reference',
    228: 'Südstrasse',
    217: 'Geyisreid',
    205: 'Ile de la Suze',
    221: 'Champagne-Piano',
    227: 'Altersheim-redenweg',
    216: 'Nidau-mittestrasse',
    211: 'Mühlefeld',
    226: 'Port Thielle',
    213: 'Port NBK',
    235: 'Altersheim-Reid',
    225: 'Swisstennis',
    207: 'Swiss Meteo Reference',
    237: 'Rolex',
    236: 'Taubenloch',
    224: 'Bözingen',
    234: 'Bözingenfeld',
    223: 'Altersheim-erlacher',
    240: 'Längholz-replacement',
    214: 'Längholz',
    208: 'Möösli',
    220: 'Altersheim-Neumarkt',
    219: 'New City',
    238: 'Museum',
    232: 'Champagne-Blumen',
    215: 'Old City',
    222: 'Spital',
    204: 'Stadtpark',
    230: 'Vingelz',
    201: 'Robert-Walser-Platz',
    233: 'Madretsch-Zukunft',
    218: 'Bahnhof',
    212: 'Seepark',
    210: 'Congresshaus',
    209: 'Madrestch-Piano',
    229: 'Viaductstrasse',
    203: 'Zentralplatz',
    231: 'Nidau-Lac',
    239: 'Nidau-wald',
    202: 'Hafen'
}
mdata, pdata = load_data()
meteo22, meteo23 = load_meteo()

def plot_maxes(data, title, save_name, highlight_stations=[201, 202, 203, 204, 205, 206, 207, 238, 218, 223, 237, 236, 215, 212], time=['2023-05', '2023-09']):
    cut_data = data.sel(time=slice(time[0], time[1]))
    maxes = cut_data.resample(time='1d').max()
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    palette = sns.color_palette("husl", len(highlight_stations))
    sns.set_palette(palette)
    # Plot settings
    plt.figure(figsize=(15, 11))

    # Plot all stations in gray
    for station in maxes.sensor:
        if station not in highlight_stations:
            maxes.sel(sensor=station).plot.line(color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for station in highlight_stations:
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        maxes.sel(sensor=station).plot.line(label=label, linewidth=1)

    # Customizing the plot
    plt.title(title)
    plt.xlabel('Hour of the Day')
    plt.ylabel('UHI Value')
    plt.grid(True)

    # Set legend outside of the plot - maybe change
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Hour of the Day', fontsize=font_size)
    plt.ylabel('Maximum Temperature', fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

def plot_mins(data, title, save_name, highlight_stations=[201, 202, 203, 204, 205, 206, 207, 238, 218, 223, 237, 236, 215, 212], time=['2023-05', '2023-09']):
    cut_data = data.sel(time=slice(time[0], time[1]))
    maxes = cut_data.resample(time='1d').min()
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    palette = sns.color_palette("husl", len(highlight_stations))
    sns.set_palette(palette)
    # Plot settings
    plt.figure(figsize=(15, 11))

    # Plot all stations in gray
    for station in maxes.sensor:
        if station not in highlight_stations:
            maxes.sel(sensor=station).plot.line(color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for station in highlight_stations:
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        maxes.sel(sensor=station).plot.line(label=label, linewidth=1)

    # Customizing the plot
    plt.title(title)
    plt.xlabel('Date')

    # Set legend outside of the plot - maybe change
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Hour of the Day', fontsize=font_size)
    plt.ylabel('Maximum Temperature', fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

title = 'Max daily temperature values'
save_name = '/home/tge/masterthesis/latek/images/new_images/data/max_daily_temp.png'
plot_maxes(mdata, title, save_name)

title = 'Min daily temperature values'
save_name = '/home/tge/masterthesis/latek/images/new_images/data/min_daily_temp.png'
plot_mins(mdata, title, save_name)
def plot_mins():
    pass

print('finished')