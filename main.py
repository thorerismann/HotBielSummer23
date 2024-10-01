from pathlib import Path
import xarray as xr
from modules import getdata, geodata
import numpy as np
from plots import boxplots, meteoplots, biasplots, uhiplots, sdtnplots
import geopandas as gpd
import pandas as pd

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
periods = {
    'summer23': ['2023-05-15', '2023-09-15'],
    'summer22': ['2022-07-09', '2022-09-10'],
    'heatwave_aug23': ['2023-08-10', '2023-08-25'],
    'heatwave_jul23': ['2023-07-07', '2023-07-12'],
    'july23': ['2023-07-01', '2023-07-31'],
    'rain_jun23': ['2023-06-28', '2023-07-01'],
}
directories = {
    'plot_directory': Path('/home/tge/master/urbanheat/charts'),
    'data_directory': Path('/home/tge/master/urbanheat/data/'),
    'data_output': Path('/home/tge/master/urbanheat/processed'),
    'appdata': Path('/home/tge/master/urbanheat/app/appdata')
}
main_uhi_highlight = [201, 202, 203, 204, 205, 238, 239, 236, 237, 232, 228]
buffers = [5, 10, 20, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]

def make_boxes(temp_data, uhi_data, time_name, time_period, directory):
    """
    Make boxplots for each sensor and each variable for the whole period and the heatwave period
    :param temp_data: temperature data
    :param uhi_data: UHI data
    :return:
    """
    # make individual station boxplots
    directory.mkdir(parents=True, exist_ok=True)
    for variable in uhi_data.data_vars:
        print(f'Plotting boxplots for {variable} during {time_period}')
        plot_directory = directory / variable
        plot_directory.mkdir(parents=True, exist_ok=True)
        boxplots.plot_hourly_boxplots(data=uhi_data, variable=variable, time_name=time_name, time=time_period,
                                      title=f'Boxplot of hourly {variable} for', y_label='Temperature (°C)',
                                      fontsize=15, save_dir=plot_directory)
    plot_directory = directory / 'temperature'
    plot_directory.mkdir(parents=True, exist_ok=True)
    boxplots.plot_hourly_boxplots_temperature(data=temp_data, time_name=time_name, time=time_period,
                                              title='Boxplot of hourly temperature ', y_label='Temperature (°C)',
                                              fontsize=15, save_dir=plot_directory)
    boxplots.plot_hourly_boxplots_temperature(data=temp_data, time_name=time_name, time=time_period,
                                              title='Boxplot of hourly temperature ', y_label='Temperature (°C)',
                                              fontsize=15, save_dir=plot_directory)


def make_geodata():
    gdc = geodata.GeoDataCollector(directories, buffers)
    gdc.save_buffered_data()

def make_meteostations_plots(meteo_data, temp_data, time_period):
    title = f'SwissMeteo AWS daily Mean, max and min air temperatures {time_period}'
    mdata = meteo_data.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    tdata = temp_data.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    meteoplots.overlay_max_mean_min_temperature(mdata, tdata,time_period,title,directories)
    title = f'SwissMeteo AWS daily mean wind speed in {time_period}'
    meteoplots.mean_windspeed(mdata, title, time_period, directories)


def make_bias_plots(meteo_data, temp_data, time_period):
    title = f'UHI Bias in {time_period}'
    bias = biasplots.calculate_bias(meteo_data, temp_data, directories)
    bias = bias.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    biasplots.plot_hourly_line(bias, directories, title=title)
    biasplots.plot_hourly_box(bias, directories, title=title)

def make_uhi_plots(uhi_data, time_period):
    title = f'UHI in Biel/Bienne, {time_period}'
    data = uhi_data.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    uhiplots.uhi_lineplot(data, main_uhi_highlight, time_period, directories, sensor_labels, title)
    uhiplots.plot_daily_max_min(data['uhi_206'], sensor_labels, title, main_uhi_highlight, time_period, directories)

def make_uhi_data(tdata, sensor_names, period):
    uhiplots.prepare_for_qgis(tdata, sensor_names, directories, period)

def make_up_close_plots(uhi_data, time_period):
    title = f'UHI in Biel/Bienne, {time_period}'
    data = uhi_data.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    uhiplots.plot_up_close(data, sensor_labels, title, main_uhi_highlight, time_period, directories)

def make_tn_sd_plots(tn, sd, time_period):
    height = 16
    width = 20
    fontsize = 20
    title_tn = f'Tropical Nights in Biel/Bienne, {time_period}'
    title_sd = f'Summer Days in Biel/Bienne, {time_period}'
    tn_sel = tn.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    sd_sel = sd.sel(time=slice(periods[time_period][0], periods[time_period][1]))
    save_name = directories['plot_directory'] / 'tn_sd'
    save_name.mkdir(parents=True, exist_ok=True)
    full_save_name_sd = save_name / f'sd_{time_period}.png'
    full_save_name_tn = save_name / f'tn_{time_period}.png'
    sdtnplots.plot_sd_tn_thresholds(sd_sel, 'summer_days', sensor_labels, full_save_name_sd, title_sd, highlight_stations=main_uhi_highlight, height=height, width=width, font_size=fontsize)
    sdtnplots.plot_sd_tn_thresholds(tn_sel, 'tropical_nights', sensor_labels, full_save_name_tn, title_tn, highlight_stations=main_uhi_highlight, height=height, width=width, font_size=fontsize)

    title_tn = f'Cumulative Tropical Nights in Biel/Bienne, {time_period}'
    title_sd = f'Cumulative Summer Days in Biel/Bienne, {time_period}'
    threshold_sd = 31
    full_save_name_sd = save_name / f'sd_cum_{threshold_sd}_{time_period}.png'
    threshold_tn = 20.4
    full_save_name_tn = save_name / f'tn_cum_{threshold_tn}_{time_period}.png'
    sdtnplots.plot_sd_tn_cumulative_sums(sd_sel, 'summer_days', threshold_sd, sensor_labels, full_save_name_sd, title_sd, highlight_stations=main_uhi_highlight, height=height, width=width, font_size=fontsize)
    sdtnplots.plot_sd_tn_cumulative_sums(tn_sel, 'tropical_nights', threshold_tn, sensor_labels, full_save_name_tn, title_tn, highlight_stations=main_uhi_highlight, height=height, width=width, font_size=fontsize)

    full_save_name_sd_excess = save_name / f'sd_excess_{time_period}.png'
    sdtnplots.plot_sd_excess_sums(sd_sel, 31, sensor_labels, save_name / f'sd_excess_{time_period}.png', 'Cumulative Summer Days Excess', highlight_stations=main_uhi_highlight, height=height, width=width, font_size=fontsize)

def load_data():
    m22, m23 = getdata.load_meteo()
    d22, d23 = getdata.load_data()
    uhi23 = getdata.make_uhi_ds(d23)
    uhi22 = getdata.make_uhi_ds(d22)
    tn = getdata.make_tropical_nights(d23, m23, np.arange(20, 25, 0.2))
    sd = getdata.make_summer_days(d23, m23, np.arange(25, 40, 1))
    sensors = getdata.load_sensor_locs()
    gdata, ludata = getdata.load_geodata()
    return m22, m23, d22, d23, uhi22, uhi23, tn, sd, sensors, gdata, ludata


def prepare_app_data():
    m22, m23, d22, d23, uhi22, uhi23, tn, sd, sensors, gdata, ludata = load_data()
    app_location = directories['appdata']
    app_location.mkdir(parents=True, exist_ok=True)

    # daily temp data at the meteo station
    a = m23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).temperature.resample(time='1d').mean()
    b = m23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).temperature.resample(time='1d').max()
    c = m23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).temperature.resample(time='1d').min()
    d = m23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).windspeed.resample(time='1d').mean()
    e = m23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).precipitation.resample(time='1d').sum()
    a.name = 'mean_temp'
    b.name = 'max_temp'
    c.name = 'min_temp'
    d.name = 'mean_wind'
    e.name = 'sum_precip'
    tempds = xr.merge([a, b, c, d, e])
    tempds.to_netcdf(app_location / 'meteo_daily.nc')

    # daily early morning UHI / city index values
    uhi_resampled = uhi23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).resample(time='1h').mean()
    uhi_4 = uhi_resampled.sel(time=uhi_resampled.time.dt.hour == 4)
    uhi_4.coords['time'] = pd.to_datetime(uhi_4.time.values).floor('D')
    # daily minmum uhi / city index values
    maximum = uhi23.sel(time=slice(periods['summer23'][0], periods['summer23'][1])).resample(time='1d').max()

    # tropical nights
    tn = tn.sel(time=slice(periods['summer23'][0], periods['summer23'][1]), threshold=20)

    # daily minimum and maximum temperatures
    maxtemp = d23.resample(time = '1d').max()
    mintemp = d23.resample(time = '1d').min()

    ds = xr.Dataset({'tropical_nights': tn.tropical_nights,
                     'max_temp': maxtemp,
                    'min_temp': mintemp,
                     'uhi_4': uhi_4.uhi_206,
                     'ci_4': uhi_4.city_index,
                     'uhi_max': maximum.uhi_206,
                     'ci_max': maximum.city_index,
                     })
    import matplotlib.pyplot as plt
    ds.sel(sensor=201).uhi_max.plot()
    plt.show()
    ds = ds.drop_duplicates('time')
    ds = ds.drop_duplicates('sensor')
    ds.to_netcdf(app_location / 'stationdata.nc')
    ds.sel(sensor=201).uhi_max.plot()
    plt.show()
    gdata_sel = gdata[gdata.buffer.isin([10,100,500])]
    ludata_sel = ludata[ludata.buffer.isin([10,100,500])]
    gdata_sel.to_csv(app_location / 'geodata.csv', index=False)
    ludata_sel.to_csv(app_location / 'landuse.csv', index=False)

    # sensors to geojson
    sensors = gpd.GeoDataFrame(sensors, geometry=gpd.points_from_xy(sensors.X, sensors.Y))
    sensors = sensors.set_crs('epsg:2056')  # Ensure CRS is set correctly
    sensors.to_file(app_location / 'sensors.geojson', driver='GeoJSON')




def main():
    # make_geodata()
    prepare_app_data()
    m22, m23, d22, d23, uhi22, uhi23, tn, sd, sensors, gdata, ludata = load_data()

    # Make boxplots
    # box_directory = directories['plot_directory'] / 'boxplots'
    # make_boxes(d23, uhi23, 'Summer 2023', periods['summer23'], box_directory)
    # make_boxes(d23, uhi23, 'August Heatwave 2023', periods['heatwave_aug23'], box_directory)
    # make_boxes(d23, uhi23, 'July Heatwave 2023', periods['heatwave_jul23'], box_directory)

    # print('Making meteo plots')
    # make_meteostations_plots(m23, d23, 'summer23')
    # make_meteostations_plots(m23, d23, 'heatwave_aug23')

    print('Making bias plots')
    # make_bias_plots(m23, d23, 'summer23')
    # make_bias_plots(m23, d23, 'heatwave_aug23')

    # print('Making UHI plots')
    # make_uhi_plots(uhi23, 'summer23')
    # make_uhi_plots(uhi23, 'heatwave_aug23')
    # make_uhi_plots(uhi23, 'heatwave_jul23')
    # make_uhi_data(d23, sensors, 'summer23')
    #
    # print('Making up close plots')
    # make_up_close_plots(uhi23['uhi_206'], 'heatwave_aug23')
    # make_up_close_plots(uhi23['uhi_206'], 'heatwave_jul23')
    # make_up_close_plots(uhi23['uhi_206'], 'rain_jun23')

    make_tn_sd_plots(tn, sd, 'summer23')




main()

