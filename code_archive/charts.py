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

def make_tropical_nights(data, thresholds=np.arange(20, 24, step=0.5)):
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

def make_summer_days(data, thresholds=list(range(25, 38))):
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
def plot_uhi23(data, title='my title', save_name = None, highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212], time=['2023-05', '2023-09']):
    cut_data = data.sel(time=slice(time[0], time[1]))
    uhi_hourly_mean = cut_data.groupby(cut_data.time.dt.hour).mean()
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
    plt.ylabel('UHI Value')
    plt.grid(True)

    # Set legend outside of the plot - maybe change
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Hour of the Day', fontsize=font_size)
    plt.ylabel('UHI Value', fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()




def plot_summer_days(data, variable='normalized', title='Summer Days Plot', ylabel=None, save_name=None,
                     highlight_stations=[201, 202, 203, 204, 205, 236, 215, 212], height=10, width=15):
    """
    Plot the summer days data with thresholds on the x-axis and color highlighting for selected stations.
    :param height:
    :param width:
    :param data: The summer days dataset, where 'threshold' is along the x-axis.
    :param title: The title of the plot.
    :param save_name: The name of the file to save the plot.
    :param highlight_stations: List of stations to highlight in color.
    """
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    # Extract the mean along the 'time' dimension to get summer days count per station for each threshold

    # Generate color palette for the highlighted stations
    palette = sns.color_palette("husl", len(highlight_stations))
    sns.set_palette(palette)

    # Plot settings
    plt.figure(figsize=(width, height))
    remove_sensors = [240, 239, 231, 214, 230]

    # Plot all stations in gray
    for station in data.sensor:
        if station not in highlight_stations:
            if station not in remove_sensors:
                data.sel(sensor=station)[variable].plot.line(x='threshold', color='gray', alpha=0.3, label='_nolegend_')

    # Highlight selected stations
    for idx, station in enumerate(highlight_stations):
        name = sensor_labels.get(station)
        label = f'Sensor {station} \n{name}'
        data.sel(sensor=station)[variable].plot.line(x='threshold', label=label, linewidth=2)

    # Customizing the plot
    plt.title(title, fontsize=font_size + 2)
    plt.xlabel('Threshold (°C)', fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.grid(True)

    # Set legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Beautify the plot
    plt.tight_layout()

    # Save the plot if a file name is provided
    if save_name:
        plt.savefig(save_name)

    # Show plot
    plt.show()


def compute_mean_and_std(data_array):
    """Calculate mean and standard deviation for a data array, rounded to 1 decimal place."""
    mean_value = round(data_array.mean().item(), 2)
    std_dev = round(data_array.std().item(), 2)
    return mean_value, std_dev
def make_summary_table(sd, tn, uhi, tnthresh, sdthresh, savename, locations):
    # Initialize the DataFrame
    summary_dfs = []
    summary_df_saves = []

    uhi_hourly_mean = uhi.uhi_206.groupby(uhi.time.dt.hour).mean()
    uhi_hourly_std = uhi.uhi_206.groupby(uhi.time.dt.hour).std()

    uhi_noon = uhi_hourly_mean.sel(hour=12).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'Mean UHI at Noon'})
    uhi_4am = uhi_hourly_mean.sel(hour=4).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'Mean UHI at 4 AM'})
    uhi_8pm = uhi_hourly_mean.sel(hour=20).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'Mean UHI at 8 PM'})

    uhi_noon_std = uhi_hourly_std.sel(hour=12).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'UHI STD at Noon'})
    uhi_4am_std = uhi_hourly_std.sel(hour=4).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'UHI STD at 4 AM'})
    uhi_8pm_std = uhi_hourly_std.sel(hour=20).to_dataframe().reset_index().set_index('sensor').drop('hour', axis=1).rename(columns={'uhi_206': 'UHI STD at 8 PM'})

    tn_sel = tn.difference_206.sel(threshold=tnthresh).to_dataframe().reset_index().set_index('sensor').drop('threshold', axis=1).rename(columns={'difference_206': 'Excess Tropical Nights'})
    sd_sel = sd.difference_206.sel(threshold=sdthresh).to_dataframe().reset_index().set_index('sensor').drop('threshold', axis=1).rename(columns={'difference_206': 'Excess Summer Days'})
    tn_ratio = tn.normalized.sel(threshold=tnthresh).to_dataframe().reset_index().set_index('sensor').drop('threshold', axis=1).rename(columns={'normalized': 'Tropical Nights Ratio'})
    sd_ratio = sd.normalized.sel(threshold=sdthresh).to_dataframe().reset_index().set_index('sensor').drop('threshold', axis=1).rename(columns={'normalized': 'Summer Days Ratio'})

    print(sd_sel)

    print(tn_sel)
    locations['sensor'] = locations['Name'].astype(int)
    locations.set_index('sensor', inplace=True)
    newdf = pd.concat([uhi_noon, uhi_4am, uhi_8pm, tn_sel, sd_sel, tn_ratio, sd_ratio, locations[['X','Y', 'Long_name']]], axis=1)
    newdf.reset_index(inplace=True)
    newdf.to_csv(savename / 'summary_table.csv', index=False)
    latekdf = newdf.set_index('sensor').copy()
    latekdf['uhi_noon_std'] = uhi_noon_std
    latekdf['uhi_4am_std'] = uhi_4am_std
    latekdf['uhi_8pm_std'] = uhi_8pm_std

    latekdf['Mean UHI at 4 AM'] = latekdf['Mean UHI at 4 AM'].round(2).astype(str)
    latekdf['Mean UHI at Noon'] = latekdf['Mean UHI at Noon'].round(2).astype(str)
    latekdf['Mean UHI at 8 PM'] = latekdf['Mean UHI at 8 PM'].round(2).astype(str)
    latekdf['UHI STD at 4 AM'] = latekdf['uhi_4am_std'].round(2).astype(str)
    latekdf['UHI STD at Noon'] = latekdf['uhi_noon_std'].round(2).astype(str)
    latekdf['UHI STD at 8 PM'] = latekdf['uhi_8pm_std'].round(2).astype(str)

    latekdf['Mean UHI at 4 AM'] = latekdf['Mean UHI at 4 AM'] + ' ± ' + latekdf['UHI STD at 4 AM']
    latekdf['Mean UHI at Noon'] = latekdf['Mean UHI at Noon'] + ' ± ' + latekdf['UHI STD at Noon']
    latekdf['Mean UHI at 8 PM'] = latekdf['Mean UHI at 8 PM'] + ' ± ' + latekdf['UHI STD at 8 PM']
    latekdf.sort_values(by='sensor', inplace=True)
    latekdf.reset_index(inplace=True)
    latekdf = latekdf[['sensor', 'Long_name', 'Mean UHI at 4 AM', 'Mean UHI at Noon', 'Mean UHI at 8 PM', 'Excess Tropical Nights','Excess Summer Days']]
    latekdf.to_latex(savename / 'summary_table.tex', index=False)

def calculate_temperature_summary(meteo22, meteo23):
    mean22 = meteo22.temperature.mean(dim='time')
    std22 = meteo22.temperature.var(dim='time')
    max22 = meteo22.temperature.max(dim='time')
    mean23 = meteo23.temperature.mean(dim='time')
    std23 = meteo23.temperature.var(dim='time')
    max23 = meteo23.temperature.max(dim='time')
    print('Summary for 2022')
    print('mean')
    print(mean22)
    print('std')
    print(std22)
    print('max')
    print(max22)
    print('Summary for 2023')
    print('mean')
    print(mean23)
    print('std')
    print(std23)
    print('max')
    print(max23)

    print('*****************************')
    print(np.max(meteo23.temperature.resample(time='1d').min().values))


def plot_heat_period(meteo23, data23, period = heatwave):
    # Slice the data according to the period
    meteo23 = meteo23.temperature.sel(time=slice(period[0], period[1]))
    data23_206 = data23.sel(time=slice(period[0], period[1]), sensor=206)
    data23_207 = data23.sel(time=slice(period[0], period[1]), sensor=207)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the three meteo stations
    plt.plot(meteo23.sel(stn='BER').time, meteo23.sel(stn='BER'), label='BER', color='blue')
    plt.plot(meteo23.sel(stn='CRM').time, meteo23.sel(stn='CRM'), label='CRM', color='green')
    plt.plot(meteo23.sel(stn='GRE').time, meteo23.sel(stn='GRE'), label='GRE', color='orange')

    # Plot the two data23 stations
    plt.plot(data23_206.time, data23_206, label='Sensor 206', linestyle='--', color='purple')
    plt.plot(data23_207.time, data23_207, label='Sensor 207', linestyle='--', color='red')

    # Add horizontal red lines at 20°C and 30°C
    plt.axhline(y=20, color='red', linestyle='-', label='20°C', linewidth=0.8)
    plt.axhline(y=30, color='red', linestyle='-', label='30°C', linewidth=0.8)

    # Set axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Reference temperatures in the August 2023 Heatwave from {} to {}'.format(period[0], period[1]))

    # Add a legend
    plt.legend(loc='upper left')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout for a cleaner look
    plt.tight_layout()

    # Show the plot
    plt.show()
def generate_plots(uhi23, tn23, sd23, locations, meteo23, meteo22, data22, data23):

    title = 'UHI at each station over the Summer period'
    time = [pd.to_datetime('2023-05'), pd.to_datetime('2023-10')]
    save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_whole_period.png'
    highlight_stations = [201, 202, 203, 204, 205, 236, 215, 212, 214, 222, 237]
    plot_uhi23(data=uhi23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)

    title = 'UHI at each station over the August Heatwave'
    time = heatwave
    save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_heatwave.png'
    plot_uhi23(data=uhi23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)

    # excess summer days
    highlight_stations = [201, 202, 203, 204, 205, 236, 215, 212, 238, 232, 222, 237]
    ylabel = 'Total summer days - Total summer days at rural reference station'
    title = 'Excess summer days May 15th to September 15th 2023'
    variable = 'difference_206'
    save_name = '/home/tge/masterthesis/latek/images/results/summer_days/summer_days_excess.png'
    plot_summer_days(data=sd23, variable=variable, title=title, ylabel=ylabel, save_name=save_name,
                     highlight_stations=highlight_stations)


    # Excess tropical nights plots
    ylabel = 'Total tropical nights - Total tropical nights at rural reference station'
    title = 'Excess Tropical Nights 15th to September 15th 2023'
    variable = 'difference_206'
    save_name = '/home/tge/masterthesis/latek/images/results/summer_days/tropical_nights_excess.png'
    plot_summer_days(data=tn23, variable=variable, title=title, ylabel=ylabel, save_name=save_name,
                     highlight_stations=highlight_stations)

    # Ratio tropical nights plots
    highlight_stations = [201, 202, 203, 204, 205, 206, 207, 236, 215, 212, 238, 232, 222, 237]
    ylabel = 'ratio of tropical nights to total days of data'
    title = 'Tropical Nights Ratio May 15th to September 15th 2023'
    variable = 'normalized'
    save_name = '/home/tge/masterthesis/latek/images/results/summer_days/tropical_nights_normalized.png'
    plot_summer_days(data=tn23, variable=variable, title=title, ylabel=ylabel, save_name=save_name,
                     highlight_stations=highlight_stations)

    # Ratio summer days plots
    highlight_stations = [201, 202, 203, 204, 205, 206, 207, 236, 215, 212, 238, 232, 222, 237]
    ylabel = 'ratio of summer days to total days of data'
    title = 'Summer Days ratio May 15th to September 15th 2023'
    variable = 'normalized'
    save_name = '/home/tge/masterthesis/latek/images/results/summer_days/summer_days_normalized.png'
    plot_summer_days(data=sd23, highlight_stations=highlight_stations, variable=variable, title=title,
                     save_name=save_name, ylabel=ylabel, height=10, width=15)

    tnthresh = 20.5
    sdthresh = 31
    save_name = Path('/home/tge/masterthesis/latek/images/results/')
    print(locations)
    make_summary_table(sd23, tn23, uhi23, tnthresh, sdthresh, save_name, locations)

    calculate_temperature_summary(meteo22, meteo23)
    plot_heat_period(meteo23, data23)

def main(plot=False):
    data23, data22 = load_data()
    meteo22, meteo23 = load_meteo()
    sensor_names = pd.read_csv(
        '/data/rawbieldata/locations/sensors_locations.csv').sort_values(by='Name')

    print('data loaded')

    uhi_23 = make_uhi_ds(data23)
    uhi_22 = make_uhi_ds(data22)

    tn_22 = make_tropical_nights(data22)
    tn_23 = make_tropical_nights(data23)

    sd_22 = make_summer_days(data22)
    sd_23 = make_summer_days(data23)

    print('data calculated')
    if plot:
        generate_plots(uhi_23, tn_23, sd_23, sensor_names, meteo23, meteo22, data22, data23)
    return uhi_22, uhi_23, tn_22, tn_23, sd_22, sd_23, sensor_names, meteo22, meteo23, data22, data23, sensor_names


    title = 'UHI at each station over the Summer period'
    time = [pd.to_datetime('2023-05'), pd.to_datetime('2023-10')]
    save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_whole_period.png'
    highlight_stations = [201, 202, 203, 204, 205, 236, 215, 212, 214, 222, 237]
    plot_uhi23(data=uhi23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)
    print('hellooooo')
    title = 'UHI at each station over the August Heatwave'
    time = heatwave
    save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_heatwave.png'
    plot_uhi23(data=uhi23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)

data23, data22 = load_data()
meteo22, meteo23 = load_meteo()
sensor_names = pd.read_csv(
    '/data/rawbieldata/locations/sensors_locations.csv').sort_values(by='Name')

print('data loaded')

uhi_23 = make_uhi_ds(data23)
# uhi_22 = make_uhi_ds(data22)
#
# tn_22 = make_tropical_nights(data22)
# tn_23 = make_tropical_nights(data23)
#
# sd_22 = make_summer_days(data22)
# sd_23 = make_summer_days(data23)
#
# title = 'UHI at each station over the Summer period'
# time = [pd.to_datetime('2023-05'), pd.to_datetime('2023-10')]
# save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_whole_period.png'
# highlight_stations = [238, 229, 215, 235, 227, 236, 224, 225, 214, 233, 209]
# plot_uhi23(data=uhi_23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)
# print('hellooooo')
# title = 'UHI at each station over the August Heatwave'
# time = heatwave
# save_name = '/home/tge/masterthesis/latek/images/results/uhi/uhi_heatwave.png'
# plot_uhi23(data=uhi_23, title=title, time=time, save_name=save_name, highlight_stations=highlight_stations)
#
#
