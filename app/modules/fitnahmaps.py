import geopandas as gpd
import pandas as pd
from folium import folium, Tooltip, CircleMarker, GeoJson
import streamlit as st
from pathlib import Path
from pyproj import Transformer
from folium.raster_layers import ImageOverlay
import streamlit_folium as stf


@st.cache_data
def load_points():
    points_path = Path(__file__).parent.parent / 'appdata' / 'sensors_locations.csv'
    points = pd.read_csv(points_path)
    points = points.dropna(subset='Name').sort_values(by='Name')
    points_gpd = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.X, points.Y))
    points = points_gpd.set_crs('epsg:2056')
    points = points.rename(columns={'Name': 'Sensor', 'Long_name': 'Name'})
    points_degrees = points.to_crs(epsg=4326)
    points_degrees = points_degrees[points_degrees['Sensor'] != 240]
    return points_degrees

def load_base_map(points, zoom_start = 13,location = [47.14093, 7.252311]):
    max_bounds = [[46.9, 6.8], [47.4, 7.6]]
    m = folium.Map(location=location, zoom_start=zoom_start)
    for _, row in points.iterrows():  # Assuming points is a Pandas DataFrame or GeoDataFrame
        CircleMarker(
            location=[row['geometry'].y, row['geometry'].x],  # Latitude, Longitude from 'geometry'
            radius=5,  # Set marker size
            color='darkblue',
            fill=True,
            fill_opacity=0.7,
            tooltip=Tooltip(f"Sensor: {row['Sensor']}, Name: {row['Name']}")  # Add a tooltip for hover
        ).add_to(m)

    return m

def overlay_image(base_map):
    """
    Adds a raster overlay to the base_map using the raster bounding box in WGS84.

    :param base_map: The Folium map object to which the raster will be added.
    :param raster_path: Path to the raster image file (e.g., .png, .jpg).
    :param name: Name of the overlay (for layer control).
    """
    image_path = str(image_paths[st.session_state.overlay['name']])
    # Get the bounding box of the raster in WGS84
    transformer = Transformer.from_crs('epsg:2056', 'epsg:4326', always_xy=True)

    # Your Swiss coordinates bounding box
    swiss_coords = [2581995.0000000000000000,1217005.0000000000000000, 2589995.0000000000000000,1225005.0000000000000000]

    # Convert the bounding box
    min_lon, min_lat = transformer.transform(swiss_coords[0], swiss_coords[1])
    max_lon, max_lat = transformer.transform(swiss_coords[2], swiss_coords[3])
    bounding_box_wgs84 = [[min_lat, min_lon], [max_lat, max_lon]]

    # Set up the base map at the center of the bounding box
    map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]


    # Overlay the raster using ImageOverlay
    ImageOverlay(
        image=image_path,
        bounds=bounding_box_wgs84,
        opacity= st.session_state.overlay['opacity'],
    ).add_to(base_map)
    stf.st_folium(base_map, height=st.session_state.overlay['height'], width=st.session_state.overlay['width'])

base = Path.cwd() / 'appdata' / 'images'
image_paths = {
    'Fitnahtemp': Path(__file__).parent.parent / 'appdata' / 'images' / 'fitnahtemp.png',
    'Fitnahuhispace': Path(__file__).parent.parent / 'appdata' / 'images' /'uhispace.png',
    'Fitnahuhistreet': Path(__file__).parent.parent / 'appdata' / 'images' / 'uhistreet.png',
    'Elevation Contours': Path(__file__).parent.parent / 'appdata' / 'images' / 'elevation_contours.png',
    'Slope': Path(__file__).parent.parent / 'appdata' / 'images' / 'slope.png'
}
legend_paths = {
    'Fitnahtemp': Path(__file__).parent.parent / 'appdata' / 'images' / 'fitnahtemp_legend.png',
    'Fitnahuhispace': Path(__file__).parent.parent / 'appdata' / 'images' /'uhi_space_legend.png',
    'Fitnahuhistreet': Path(__file__).parent.parent / 'appdata' / 'images' /'uhis_treet_legend.png'
}
def select_overlay():
    with st.form('Choose Overlay'):
        overlays = ['Fitnahtemp', 'Fitnahuhispace', 'Fitnahuhistreet', 'Elevation Contours', 'Slope']
        overlay = st.selectbox('Select an overlay:', overlays)
        height = st.number_input('Enter the height of the overlay:', min_value=500, max_value=2000, value=500)
        width = st.number_input('Enter the width of the overlay:', min_value=500, max_value=2000, value=500)
        opacity = st.slider('Select the opacity of the overlay:', min_value=0.0, max_value=1.0, value=0.7)
        overlay_button = st.form_submit_button('Add Overlay')
    if overlay_button:
        st.session_state['overlay'] = {
            'name': overlay,
            'height': height,
            'width': width,
            'opacity': opacity,
        }

def main():
    points = load_points()
    map = load_base_map(points)
    select_overlay()
    meta = st.session_state.get('overlay')
    if meta:
        st.write(f"Displaying {st.session_state.overlay['name']} with default parameters")
        overlay_image(map)
    else:
        st.write("Displaying Slope with default parameters")
        st.session_state['overlay'] = {
            'name': 'Slope',
            'height': 500,
            'width': 500,
            'opacity': 0.7,
        }
        overlay_image(map)

