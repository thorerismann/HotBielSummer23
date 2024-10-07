import holoviews as hv
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


def select_plots(data):
    with st.container(border=True):
        st.subheader('Time series visualization')
        st.info('Data available in daily format due to space constraints in the app. See the github repository for 10-minute and hourly data')
        with st.form('plot_data'):
            st.subheader('Plot Station Data')
            st.info('Select a data type and any number of stations to plot.')
            st.info('ci_4 = City Index at 4 AM, uhi_4 = UHI at 4 AM, ')
            dtype  = st.selectbox('select data type',['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])
            stations = st.multiselect('Select stations', list(range(201, 240)))
            addmeteo_max = st.checkbox('Add Meteo Max')
            addmeteo_min = st.checkbox('Add Meteo Min')
            plot_stations = st.form_submit_button('Plot Stations')
        if plot_stations:
            if len(stations) > 0:
                main_plot = data.sel(sensor=stations)[dtype].to_dataframe().reset_index().hvplot(by='sensor', y=dtype, x='time', width=600, height=400, legend='top_left')
                if addmeteo_max:
                    meteomax = data['meteo_max'].hvplot(label='Meteo Station Max', color='red')
                    main_plot = main_plot * meteomax
                if addmeteo_min:
                    meteomin = data['meteo_min'].hvplot(label='Meteo Station Min', color='red')
                    main_plot = main_plot * meteomin

                st.bokeh_chart(hv.render(main_plot))


def plot_overall_group(station):
    with st.container(border=True):
        st.subheader('Plot pre-defined groups')
        st.json(station_dict, expanded=False)
        st.info('Select a data type and toggle Plot Groups below to see data grouped by features (expand above to see the groups)')
        dtype = st.selectbox('select data type',['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])
        plot_dict = {}
        overall_dict = {}
        for group, stations in station_dict.items():
            overall_dict[group] = station.sel(sensor=stations)[dtype].mean(dim='sensor')
        ds = xr.Dataset(overall_dict)
        st.bokeh_chart(hv.render(ds.hvplot.line(legend='top_left')))
