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


mdata, pdata = load_data()

def plot_forest_stations(data, path='/home/tge/masterthesis/latek/images/new_images/data/forest.png'):
    color_map = {214: 'red', 239: 'blue', 240: 'green'}
    stations = [214, 239, 240]
    maxes = data.sel(sensor = stations).resample(time='1d').max()
    mins = data.sel(sensor = stations).resample(time='1d').min()
    means = data.sel(sensor = stations).resample(time='1d').mean()
    plt.figure(figsize=(12, 6))
    for station in stations:
        maxes.sel(sensor=station).plot.line(hue='sensor', label=f'Max {station}', color=color_map[station])
        mins.sel(sensor=station).plot.line(hue='sensor', label=f'Min {station}', color=color_map[station], linestyle='dashed')
        means.sel(sensor=station).plot.line(hue='sensor', label=f'Mean {station}', color=color_map[station], linestyle='dotted')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

plot_forest_stations(mdata)

# uncomment for single plots of each missing data
# mdata.sel(sensor = 230).plot()
# plt.show()
# mdata.sel(sensor = 231).plot()
# plt.show()
# mdata.sel(sensor = 239).plot()
# plt.show()
# mdata.sel(sensor = 214).plot()
# plt.show()
print('stop')

stations = list(range(231, 240))
mdata.sel(sensor=stations).plot.line(hue='sensor')
plt.show()

