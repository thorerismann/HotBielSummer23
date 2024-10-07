from pathlib import Path

import streamlit as st
import streamlit_folium as stf
import xarray as xr

from modules import about, fitnahmaps, dataviz, stationmap

base_path = Path(__file__).parent
data_path = base_path / 'appdata'
image_path = data_path / 'images'
favicon = base_path / 'favicon.ico'

@st.cache_data()
def load_data(basepath=data_path / 'app_data.nc'):
    with xr.open_dataset(basepath) as ds:
        loaded = ds.load()
    return ds


maps_paths = {
    'elevation': image_path / 'elevation_contours.png',
    'fitnahtemp': image_path / 'fitnahtemp.png',
    'slope': image_path / 'slope.png',
    'uhistreet': image_path / 'uhistreet.png',
    'uhispace': image_path / 'uhispace.png',
}

legends_paths = {
    'uhistreet': image_path / 'uhi_street_legend.png',
    'uhispace': image_path / 'uhi_space_legend.png',
    'fitnahtemp': image_path / 'fitnahtemp_legend.png',
}

st.set_page_config(
    page_title='Hot Biel Summer',
    page_icon=str(favicon),
    layout='centered',
    initial_sidebar_state='auto'
)

def main():


    data = load_data()
    activities = ['Home', 'Fitnah Maps', 'Time Series', 'Groups', 'BAMBI Model']
    choice = st.sidebar.selectbox('Select Activity', activities)
    if choice == 'Home':
        st.markdown("""
        # Hot Biel Summer

        Explore the 2023 urban temperature data from Biel.

        You can view some summary results of the data from the **2023 measurement campaign** below. Select the menu on the left to:

        - **View Geospatial Data Layers**: Explore various geographical layers and overlays.
        - **Explore Station Data as Time Series**: Analyze temperature trends over time from individual stations.
        - **Build Your Own Model**: Use the **Bayesian Model Building Interface (BAMBI)** to create your own predictions based on the data.
        """)
        stationmap.select_plotting(data)
    elif choice == 'BAMBI Model':
        st.write('coming soon')
    elif choice == 'About':
        about.display_information()
    elif choice == 'Time Series':
        dataviz.select_plots(data)
    elif choice == 'Groups':
        dataviz.plot_overall_group(data)
    elif choice == 'Fitnah Maps':
        basemap = fitnahmaps.MapDisplay.load_base_map(fitnahmaps.MapDisplay.load_points())
        fitnahmaps.select_overlay()
        meta = st.session_state.get('overlay')
        if meta:
            newmap = fitnahmaps.MapDisplay.overlay_image(basemap)
            stf.st_folium(newmap, height=st.session_state.overlay['height'], width=st.session_state.overlay['width'])
            if meta['legend']:
                st.image(str(meta['legend']), caption="Legend", width=300)
main()
