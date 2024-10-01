from pathlib import Path

import streamlit as st
import streamlit_folium as stf

from modules import about, maps, dataviz

st.set_page_config(layout="wide")

def main():
    st.title('Welcome to Biel Heatmaps')
    activities = ['Home', 'Maps', 'Data Viz', 'Station Explorer', 'BAMBI Model']
    choice = st.sidebar.selectbox('Select Activity', activities)


    if choice == 'Home':
        basemap = maps.MapDisplay.load_base_map(maps.MapDisplay.load_points())
        stf.st_folium(basemap)
    elif choice == 'BAMBI Model':
        st.write('coming soon')
    elif choice == 'About':
        about.display_information()
    elif choice == 'Data Viz':
        dataviz.main()
    elif choice == 'Maps':
        basemap = maps.MapDisplay.load_base_map(maps.MapDisplay.load_points())
        maps.select_overlay()
        meta = st.session_state.get('overlay')
        if meta:
            newmap = maps.MapDisplay.overlay_image(basemap)
            stf.st_folium(newmap, height=st.session_state.overlay['height'], width=st.session_state.overlay['width'])
            if meta['legend']:
                st.image(str(meta['legend']), caption="Legend", width=300)
main()
