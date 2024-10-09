from pathlib import Path

import streamlit as st
import xarray as xr

from modules import about, fitnahmaps, dataviz, stationmap


# Set Paths
base_path = Path(__file__).parent
data_path = base_path / 'appdata'
image_path = data_path / 'images'
favicon = base_path / 'favicon.ico'

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


# function to load main dataset
@st.cache_data()
def load_data(basepath=data_path / 'app_data.nc'):
    with xr.open_dataset(basepath) as ds:
        loaded = ds.load()
    return ds


# set page configuration
st.set_page_config(
    page_title='Hot Biel Summer',
    page_icon=str(favicon),
    layout='wide',
    initial_sidebar_state='auto'
)

def main():
    data = load_data()
    activities = ['Home', 'Fitnah Maps', 'Station Maps', 'Time Series', 'Groups', 'BAMBI Model', 'References & Acknowledgements']
    with st.container(border=True):
        st.markdown('##### Choose Activity')
        choice = st.selectbox('choose activity', activities)
    if choice == 'Home':
        st.markdown("""
        # Hot Biel Summer
        
        Data visualization tool for the 2023 Urban Heat Island measurement campaign in Biel.
        
        Contact via thor @ hammerdirt . ch or better yet make a comment on the repository (button above).

        **Select from the menu above to get started**
        - Fitnah Maps - Map of the UHI model of Biel provided by the canton
        - Station Maps - Map of the results of the measurement campaign
        - Time Series - Plot timeseries of the different data variables
        - Groups - Plot timeseries of predefined station groups
        - References - Packages used, inspiration
        """)

    elif choice == 'BAMBI Model':
        st.write('coming soon')
    elif choice == 'About':
        about.display_information()
    elif choice == 'Station Maps':
        stationmap.select_plotting(data.drop_sel(sensor=[240, 231]))
    elif choice == 'Time Series':
        dataviz.select_plots(data)
    elif choice == 'Groups':
        dataviz.plot_overall_group(data)
    elif choice == 'Fitnah Maps':
        st.markdown('#### View Fitnah Maps')
        fitnahmaps.main()
    elif choice == 'References & Acknowledgements':
        st.markdown("""
        ##### Packages used
        
        - QGIS, Folium and Streamlit-Folium to create the FITNAH maps
        - xarray / pandas for data manipulation
        - pydeck and geopandas to create the station maps
        - streamlit linecharts to display the station data
        - streamlit api to write this app, of course
        
        ##### Support and Help
        - Ville de Bienne for making the project run smoothly re authorizations and support.
        - Geographisches Institut UNIBE for everything: the sensors, the idea, sparking my interest in the UHI to begin with.
        - hammerdirt for advice, support, feedback when no one else takes a look
        """)
main()
