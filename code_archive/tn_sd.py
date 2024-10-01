
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
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
whole_period = ['2023-05-16', '2023-09-16']
heatwave = ['2023-08-10', '2023-08-25']
sensor_names = pd.read_csv('/data/rawbieldata/locations/sensors_locations.csv').sort_values(by='Name')
sensor_names = sensor_names.rename(columns={'Name': 'sensor'})



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
    mdata.loc[dict(sensor=231,time=slice('2023-05-15', '2023-06-20'))] = np.nan
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


def make_tropical_nights(mydata, period, thresholds, pstring):
    """
    Make tropical nights dataset, normalized by the number of non-NaN days per sensor, and compare sensors 206 and 207.
    :param data: Input dataset with temperature data.
    :param thresholds: List of threshold temperatures to count tropical nights.
    :return: Dataset with absolute tropical nights, normalized tropical nights, and differences between sensors.
    """
    results_absolute = []
    results_normalized = []
    results_206 = []
    results_207 = []
    data = mydata.sel(time=slice(period[0], period[1]))

    for threshold in thresholds:
        # Resample daily minimum temperature
        dsmin = data.resample(time='1d').min()

        # Create a mask for non-NaN values
        valid_days = dsmin.notnull().sum(dim='time')  # Count of non-NaN days for each sensor

        # Calculate tropical nights where the min temperature exceeds the threshold
        tn = dsmin > threshold
        tnights = tn.sum(dim='time')  # Count of tropical nights for each sensor


        # Subtract the tropical nights, but only for the overlapping periods
        difference_206 = tn.sum(dim='time') - tn.sel(sensor=206).sum(dim='time')

        difference_207 = tn.sum(dim='time') - tn.sel(sensor=207).sum(dim='time')

        # Normalize the tropical nights by the number of valid (non-NaN) days
        tnights_normalized = tnights / valid_days

        # Add threshold as a dimension to both absolute and normalized results
        tnights_abs_expanded = tnights.expand_dims(threshold=[threshold])
        tnights_norm_expanded = tnights_normalized.expand_dims(threshold=[threshold])

        # Append results
        results_absolute.append(tnights_abs_expanded)
        results_normalized.append(tnights_norm_expanded)
        results_206.append(difference_206)
        results_207.append(difference_207)

    # Concatenate along the new threshold dimension for absolute and normalized results
    tropical_nights_absolute = xr.concat(results_absolute, dim='threshold')
    tropical_nights_normalized = xr.concat(results_normalized, dim='threshold')
    difference_206 = xr.concat(results_206, dim='threshold')
    difference_207 = xr.concat(results_207, dim='threshold')

    # Combine both datasets into a single dataset with two variables
    result = xr.Dataset({
        'absolute': tropical_nights_absolute,
        'normalized': tropical_nights_normalized,
        'difference_206': difference_206,
        'difference_207': difference_207
    })

    return result


def make_summer_days(mydata, period, thresholds, pstring):
    """
    Make summer days dataset, normalized by the number of non-NaN days per sensor.
    :param data: Input dataset with temperature data.
    :param thresholds: List of threshold temperatures to count summer days.
    :return: Dataset with absolute summer days and normalized summer days.
    """
    results_absolute = []
    results_normalized = []
    results_206 = []
    results_207 = []
    data = mydata.sel(time=slice(period[0], period[1]))
    for threshold in thresholds:
        # Resample daily maximum
        dsmax = data.resample(time='1d').max()

        # Create a mask for non-NaN values
        valid_days = dsmax.notnull().sum(dim='time')  # Count of non-NaN days for each sensor

        # Calculate summer days where the max temperature exceeds the threshold
        sd = dsmax > threshold
        sdays = sd.sum(dim='time')  # Count of summer days for each sensor

        # Subtract the summer days, but only for the overlapping periods
        difference_206 = sdays - sdays.sel(sensor=206)

        difference_207 = sdays - sdays.sel(sensor=206)


        # Normalize the summer days by the number of valid (non-NaN) days
        sdays_normalized = sdays / valid_days

        # Add threshold as a dimension to both absolute and normalized results
        sdays_abs_expanded = sdays.expand_dims(threshold=[threshold])
        sdays_norm_expanded = sdays_normalized.expand_dims(threshold=[threshold])

        # Append results
        results_absolute.append(sdays_abs_expanded)
        results_normalized.append(sdays_norm_expanded)
        results_206.append(difference_206)
        results_207.append(difference_207)

    # Concatenate along the new threshold dimension for absolute and normalized results
    summer_days_absolute = xr.concat(results_absolute, dim='threshold')
    summer_days_normalized = xr.concat(results_normalized, dim='threshold')
    difference_206 = xr.concat(results_206, dim='threshold')
    difference_207 = xr.concat(results_207, dim='threshold')

    # Combine both datasets into a single dataset with two variables
    result = xr.Dataset({
        'absolute': summer_days_absolute,
        'normalized': summer_days_normalized,
        'difference_206': difference_206,
        'difference_207': difference_207
    })

    return result

def prepare_qgis(tns, thresh, sensor_names, pstring):
    tn = tns.sel(threshold=thresh, sensor= [x for x in np.unique(tns.sensor) if x != 240]).to_dataframe().reset_index()
    sensor_names = sensor_names[sensor_names.sensor != 240]
    tn_xy = tn.merge(sensor_names, on='sensor')
    tn_xy.to_csv(f'/home/tge/masterthesis/latek/images/new_images/results/tn_{pstring}_{thresh}.csv', index=False)
    return tn_xy

def prepare_qgis_sd(sds, thresh, sensor_names, pstring):
    sd = sds.sel(threshold=thresh, sensor= [x for x in np.unique(sds.sensor) if x != 240]).to_dataframe().reset_index()
    sensor_names = sensor_names[sensor_names.sensor != 240]
    sd_xy = sd.merge(sensor_names, on='sensor')
    sd_xy.to_csv(f'/home/tge/masterthesis/latek/images/new_images/results/sd_{pstring}_{thresh}.csv', index=False)
    return sd_xy



mdata, pdata = load_data()
period = heatwave
thresholds = list(np.arange(20, 24.5, 0.5))
pstring = 'heatwave'
tn23 = make_tropical_nights(mdata, period, thresholds, pstring)

prepare_qgis(tn23, 20.5, sensor_names, pstring)
prepare_qgis(tn23, 20, sensor_names, pstring)
threshold = 31
period = whole_period
pstring = 'whole'
thresholds = list(range(25, 38))
sd23 = make_summer_days(mdata, period, thresholds, pstring)
prepare_qgis_sd(sd23, 31, sensor_names, pstring)

def plot_cumulative_sd_tn(mydata, period, tsd=30, ttn = 20, width=15, height=10, highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212]):
    """
    Plot the cumulative sum of summer days (temperature > threshold) for all sensors minus sensor 206.
    :param mydata: Input dataset with temperature data.
    :param period: Time period to analyze.
    :param threshold: Threshold temperature to count summer days (default 30°C).
    """
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    # Extract the mean along the 'time' dimension to get summer days count per station for each threshold

    # Generate color palette for the highlighted stations
    palette = sns.color_palette("husl", len(highlight_stations))
    sns.set_palette(palette)

    # Plot settings
    plt.figure(figsize=(width, height))
    remove_sensors = [240]
    # Select data for the specified period
    data = mydata.sel(time=slice(period[0], period[1]))

    # Resample daily maximum temperature per day
    dsmax = data.resample(time='1d').max()
    dsmin = data.resample(time='1d').min()

    # Calculate summer days where the max temperature exceeds the threshold
    summer_days = dsmax > tsd
    tropical_nights = dsmin > ttn
    sd_csum = summer_days.cumsum(dim='time') - summer_days.sel(sensor=206).cumsum(dim='time')
    tn_csum = tropical_nights.cumsum(dim='time') - tropical_nights.sel(sensor=206).cumsum(dim='time')
    sd_csum.name = 'summer_days'
    print(sd_csum)
    tn_csum.name = 'tropical_nights'
    print(tn_csum)
    plt.figure(figsize=(width, height))
    for station in sd_csum.sensor:
        if station not in highlight_stations:
            if station not in remove_sensors:
                sd_csum.sel(sensor=station).plot.line(color='gray', label='_nolegend_')
    # Highlight selected stations
    for idx, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        sd_csum.sel(sensor=station).plot.line(x='time', label=label, linewidth=2)

    # Customizing the plot
    plt.title('Cumulative sum of excess summer days for each station', fontsize=font_size + 2)
    plt.xlabel('time', fontsize=font_size)
    plt.ylabel('Excess summer days', fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Beautify the plot
    plt.tight_layout()

    # Save the plot if a file name is provided
    plt.savefig('/home/tge/masterthesis/latek/images/new_images/results/cumulative_summer_days.png')

    # Show plot
    plt.show()

    plt.figure(figsize=(width, height))
    for station in tn_csum.sensor:
        if station not in highlight_stations:
            if station not in remove_sensors:
                tn_csum.sel(sensor=station).plot.line(color='gray', label='_nolegend_')
    # Highlight selected stations
    for idx, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        tn_csum.sel(sensor=station).plot.line(x='time', label=label, linewidth=2)

    # Customizing the plot
    plt.title('Cumulative sum of Tropical Nights for each station', fontsize=font_size + 2)
    plt.xlabel('time', fontsize=font_size)
    plt.ylabel('Tropical Nights', fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Beautify the plot
    plt.tight_layout()

    # Save the plot if a file name is provided
    plt.savefig('/home/tge/masterthesis/latek/images/new_images/results/cumulative_tropical_nights.png')

    # Show plot
    plt.show()

    return sd_csum, tn_csum
period = whole_period
highlight_stations = [201, 202, 203, 204, 205, 236, 215, 212, 238, 239, 214, 208, 218, 232]
plot_cumulative_sd_tn(mdata, period, tsd=30, ttn = 20, width=15, height=10, highlight_stations=highlight_stations)

print('hey there')