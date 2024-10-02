import holoviews as hv
import pandas as pd
import hvplot.pandas
import hvplot.xarray
from pathlib import Path
import xarray as xr
import streamlit as st

meteo_path = Path(__file__).parent.parent / 'appdata' /'meteo_daily.nc'
station_path = Path(__file__).parent.parent / 'appdata' /'stationdata.nc'

station_dict = {'rural': [206, 207],
                'care_homes': [220, 222, 223, 235, 227],
                'lake': [231, 230, 212, 202],
                'jura': [222, 235],
                'suze': [238, 236, 224, 203, 202, 213],
                'water':[231, 230, 212, 202, 238, 236, 224, 203, 202, 213],
                'forest': [214, 239],
                'boujean': [237, 234, 225],
                'flat':[209, 233, 210, 220, 218, 201, 229, 238, 215, 219, 204, 232, 221, 224, 227, 236, 225, 237, 234, 228],
                'nidauish':[239, 216, 226, 213, 211, 208, 239]
                }

@st.cache_data
def load_data():
    with xr.open_dataset(meteo_path) as ds:
        meteo = ds.load()
    with xr.open_dataset(station_path) as ds:
        station = ds.load()
    return meteo.to_dataframe(), station.to_dataframe()


def select_plots(meteo, station):
    with st.form('plot_data'):
        st.subheader('Plot Station Data')
        st.info('Select a data type and any number of stations to plot.')
        dtype  = st.selectbox('select data type',['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])
        stations = st.multiselect('Select stations', list(range(201, 240)))
        addmeteo_max = st.checkbox('Add Meteo Max')
        addmeteo_min = st.checkbox('Add Meteo Min')
        plot_stations = st.form_submit_button('Plot Stations')
    if plot_stations:
        if len(stations) > 0:
            main_plot = station.to_xarray().sel(sensor=stations)[dtype].hvplot(by='sensor', y=dtype, width=810, height=400, legend='top_left')
            if addmeteo_max:
                meteomax = meteo.to_xarray()['max_temp'].sel(stn='GRE').hvplot(y='max_temp', width=810, height=400, label='Meteo Station Max', color='red')
                main_plot = main_plot * meteomax
            if addmeteo_min:
                meteomin = meteo.to_xarray()['min_temp'].sel(stn='GRE').hvplot(y='min_temp', width=810, height=400, label='Meteo Station Min', color='red')
                main_plot = main_plot * meteomin
            st.bokeh_chart(hv.render(main_plot))

def plot_groups(station):
    with st.container(border=True):
        st.subheader('Plot Groups')
        st.json(station_dict, expanded=False)
        st.info('Select a data type and toggle Plot Groups below to see data grouped by features (expand above to see the groups)')
        dtype = st.selectbox('select data type',['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])
        plot_dict = {}
        overall_dict = {}
        if st.toggle('Plot Groups'):
            for group, stations in station_dict.items():
                plot_dict[group] = hv.render(station.to_xarray().sel(sensor=stations)[dtype].hvplot(by='sensor', y=dtype, width=500, height=300, legend='top_left', title=group))
                overall_dict[group] = station.to_xarray().sel(sensor=stations)[dtype].mean(dim='sensor')
            ds = xr.Dataset(overall_dict)
            st.bokeh_chart(hv.render(ds.hvplot.line(width=800, height=400, legend='top_left')))
            c1, c2 = st.columns(2, gap='small')
            for index, (group, plot) in enumerate(plot_dict.items()):
                if index % 2 == 0:
                    c1.bokeh_chart(plot)
                else:
                    c2.bokeh_chart(plot)

def main():
    meteo, station = load_data()
    st.subheader('Time series visualization')
    st.info('Data available in daily format due to space constraints in the app. See the github repository for 10-minute and hourly data')
    plot_groups(station)
    select_plots(meteo, station)



