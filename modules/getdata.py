import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def load_data():
    """
    Load the main and pilot data and filter out the 'ref' sensor.

    :return:
    """
    with xr.open_dataset(Path('/home/tge/master/urbanheat/data/rawbieldata/main_data.nc')) as ds:
        mdata = ds.load()
    mdata = mdata.rename({'logger': 'sensor'}).temperature
    with xr.open_dataset(Path('/home/tge/master/urbanheat/data/rawbieldata/pilot_data.nc')) as ds:
        pdata = ds.load()

    mdata.loc[dict(sensor=231,time=slice('2023-05-15', '2023-06-03'))] = np.nan
    mdata.loc[dict(sensor=[238, 233, 205, 209, 211, 219],time=slice('2023-05-15', '2023-05-17'))] = np.nan
    sensors = pdata.sensor.values
    sensors = [s for s in sensors if s != 'ref']
    pdata = pdata.sel(sensor=sensors)
    sensors = [int(s) for s in pdata.sensor]
    pdata['sensor'] = sensors
    # Filter out 'ref' coordinate and its data
    pdata = pdata.temp
    pdata.name = 'temperature'
    return pdata, mdata


def load_meteo():
    df = pd.read_table('/home/tge/master/urbanheat/data/meteo/all_21_23.txt', sep=r'\s+', skiprows=1, dtype=str)
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

def make_uhi_ds(data):
    """
    Make UHI dataset from main and pilot data

    :param data:
    :return: dataset
    """
    ds = data.sel(time=slice('2023-05-15', '2023-09-15'))
    city_index = ds - ds.mean(dim='sensor')
    uhi_207 = ds - ds.sel(sensor=207)
    uhi_206 = ds - ds.sel(sensor=206)

    ds = xr.Dataset({'city_index': city_index, 'uhi_207': uhi_207, 'uhi_206': uhi_206})
    return ds

def load_sensor_locs():
    sensor_names = pd.read_csv('/home/tge/master/urbanheat/data/rawbieldata/locations/sensors_locations.csv').sort_values(by='Name')
    sensor_names = sensor_names.rename(columns={'Name': 'sensor'})
    return sensor_names


def load_geodata():
    buffered = pd.read_csv('/home/tge/master/urbanheat/processed/geo_data/buffered_data.csv')
    buffered = buffered.rename(columns={'logger': 'sensor'})
    sensors = load_sensor_locs()
    data = buffered.merge(sensors, on='sensor')

    buffered_lu = pd.read_csv('/home/tge/master/urbanheat/processed/geo_data/buffered_landuse.csv')
    buffered_lu = buffered_lu.rename(columns={'logger': 'sensor'})
    data2 = buffered_lu.merge(sensors, on='sensor')
    return data, data2


def make_tropical_nights(data, meteo, thresholds=np.arange(20, 24, step=0.5)):
    """
    Make tropical nights dataset, normalized by the number of non-NaN days per sensor, and compare sensors 206 and 207.
    :param data: Input dataset with temperature data.
    :param thresholds: List of threshold temperatures to count tropical nights.
    :return: Dataset with absolute tropical nights, normalized tropical nights, and differences between sensors.
    """
    tns = []
    for threshold in thresholds:
        # Resample daily minimum temperature
        dsmin = data.resample(time='1d').min()


        # Calculate tropical nights where the min temperature exceeds the threshold
        tn = dsmin > threshold

        tnights = tn.expand_dims(threshold=[threshold])

        # Append results
        tns.append(tnights)

    # Concatenate along the new threshold dimension for absolute and normalized results
    trop_night = xr.concat(tns, dim='threshold')
    trop_night.name = 'tropical_nights'

    meteo_sel = meteo.sel(stn='GRE').reindex(time=trop_night.time).drop_vars('stn').squeeze()
    maxtemp = meteo_sel.temperature.resample(time='1d').max()
    mintemp = meteo_sel.temperature.resample(time='1d').min()
    maxtemp_prev = maxtemp.shift(time=1)

    ds = xr.Dataset({'tropical_nights': trop_night, 'maxtemp': maxtemp, 'mintemp': mintemp, 'maxtemp_prev': maxtemp_prev})
    return ds

def make_summer_days(data, meteo, thresholds=np.arange(25, 40, step=1)):
    """
    Make summer days dataset, normalized by the number of non-NaN days per sensor, and compare sensors 206 and 207.
    :param meteo: meteo dataset
    :param data: Input dataset with temperature data.
    :param thresholds: List of threshold temperatures to count summer days.
    :return: Dataset with absolute summer days, normalized summer days, and differences between sensors.
    """
    sds = []

    for threshold in thresholds:
        # Resample daily maximum temperature
        dsmax = data.resample(time='1d').max()


        sd = dsmax > threshold

        sdays = sd.expand_dims(threshold=[threshold])

        # Append results
        sds.append(sdays)

    # Concatenate along the new threshold dimension for absolute and normalized results
    summer_days = xr.concat(sds, dim='threshold')
    summer_days.name = 'summer_days'

    meteo_sel = meteo.sel(stn='GRE').reindex(time=summer_days.time).drop_vars('stn').squeeze()
    maxtemp = meteo_sel.temperature.resample(time='1d').max()
    mintemp = meteo_sel.temperature.resample(time='1d').min()
    maxtemp_prev = maxtemp.shift(time=1)
    ds = xr.Dataset(
        {'summer_days': summer_days, 'maxtemp': maxtemp, 'mintemp': mintemp, 'maxtemp_prev': maxtemp_prev})
    return ds
