from pathlib import Path
import xarray as xr
import streamlit as st
from datetime import datetime
import pandas as pd

meteo_path = Path(__file__).parent.parent / 'appdata' /'meteo_daily.nc'
station_path = Path(__file__).parent.parent / 'appdata' /'stationdata.nc'

station_dict = {'rural': [206, 207],
                'care_homes': [220, 222, 223, 235, 227],
                'lake': [231, 230, 212, 202],
                'jura': [222, 235],
                'forest': [214, 239],
                'boujean': [237, 234, 225],
                'flat':[209, 233, 210, 220, 218, 201, 229, 238, 215, 219, 204, 232, 221, 224, 227, 236, 225, 237, 234, 228],
                'nidauish':[239, 216, 226, 213, 211, 208, 239]
                }




def plot_overall_group(station):
    st.subheader('Plot Pre-defined Groups')
    st.json(station_dict, expanded=False)
    st.info('Select a data type and toggle Plot Groups below to see data grouped by features.')

    # Select the data type
    dtype = st.selectbox('Select data type', ['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])

    overall_dict = {}
    for group, stations in station_dict.items():
        overall_dict[group] = station.sel(sensor=stations)[dtype].mean(dim='sensor')

    # Convert to pandas DataFrame for easier plotting with st.line_chart
    ds_df = xr.Dataset(overall_dict).to_dataframe()

    # Create a line chart with Streamlit's line_chart for better mobile optimization
    st.line_chart(ds_df)


def select_plots(data):
    st.subheader('Time series visualization')
    st.info('Data available in daily format due to space constraints in the app. See the github repository for 10-minute and hourly data')
    with st.form('plot_data'):
        st.subheader('Plot Station Data')
        st.info('Select a data type and any number of stations to plot.')
        st.info('ci_4 = City Index at 4 AM, uhi_4 = UHI at 4 AM, ')
        dtype  = st.selectbox('select data type',['uhi_4', 'ci_4', 'tropical_nights', 'max_temp', 'min_temp', 'uhi_max', 'ci_max'])
        stations = list(range(201, 241))
        with st.expander('Select stations', expanded=False):
            cols = st.columns(2)
            for i, station in enumerate(stations):
                col = cols[i % 2]
                with col:
                    st.checkbox(f'Station {station}', key=f'station_{station}', value=False)
        start = st.date_input('Select Start', value = datetime.strptime('2023-05-15', '%Y-%m-%d'), min_value=datetime.strptime('2023-05-15', '%Y-%m-%d'), max_value=datetime.strptime('2023-09-15', '%Y-%m-%d'))
        end = st.date_input('Select End', value = datetime.strptime('2023-09-15', '%Y-%m-%d'), min_value=datetime.strptime('2023-05-15', '%Y-%m-%d'), max_value=datetime.strptime('2023-09-15', '%Y-%m-%d'))
        plot_stations = st.form_submit_button('Plot Stations')
    if plot_stations:
        selection = {k:v for k,v in st.session_state.items() if 'station_' in k}
        selected_stations = [k for k,v in selection.items() if v]
        if len(selected_stations) < 1:
            st.warning('Select at least one station to plot data')
            return
        if end <= start:
            st.warning('End date must be greater than start date')
            return
            # Select data for the selected stations and data type
        stations = [int(x[-3:]) for x in selected_stations]
        st.session_state['plot_stations_data'] = {
            'stations': stations,
            'dtype': dtype,
            'start': pd.to_datetime(start),
            'end': pd.to_datetime(end),
        }
    meta = st.session_state.get('plot_stations_data')

    if meta:
        selected_data = data.sel(sensor=meta['stations'], time=slice(meta['start'], meta['end']))[meta['dtype']].to_dataframe().reset_index()

        pivoted_data = selected_data.pivot(index='time', columns='sensor', values=meta['dtype'])

        # Plot using Streamlit's native line chart
        st.line_chart(pivoted_data)

    else:
        st.write('Displaying default: uhi at 4 am between june 15th and aug 15th')
        selected_data = data.sel(sensor=[201, 202, 203], time=slice('2023-06-15', '2023-08-15'))['uhi_4'].to_dataframe().reset_index()

        pivoted_data = selected_data.pivot(index='time', columns='sensor', values='uhi_4')

        # Plot using Streamlit's native line chart
        st.line_chart(pivoted_data)
