import streamlit as st
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as colors



def prep_data(dataset, selection):
    if selection == 'Max Temp':
        dataframe = dataset.max_temp.max(dim='time').to_dataframe()
        colname = 'max_temp'
    if selection == 'Min Temp':
        dataframe = dataset.min_temp.min(dim='time').to_dataframe()
        colname = 'min_temp'
    if selection == 'City Index':
        dataframe = dataset.ci_max.mean(dim='time').to_dataframe()
        colname = 'ci_max'
    if selection == '4AM UHI':
        dataframe = dataset['uhi_4'].mean(dim='time').to_dataframe()
        colname = 'uhi_4'
    if selection == 'FITNAH Prediction':
        dataframe = dataset['fitnah_temp'].sel(buffer=50).drop_vars('buffer').to_dataframe()
        colname = 'fitnah_temp'
    dataframe['X'] = dataset['X'].values
    dataframe['Y'] = dataset['Y'].values
    norm = colors.Normalize(vmin=dataframe[colname].min(), vmax=dataframe[colname].max())

    colormap = cm.get_cmap('coolwarm')  # You can choose another colormap like 'viridis', 'plasma', etc.

    # Apply the colormap to get RGB values
    dataframe['color'] = dataframe[colname].apply(lambda x: colormap(norm(x)))

    # Convert the RGBA colormap output into a format pydeck can use (integers between 0-255)
    dataframe['color'] = dataframe['color'].apply(lambda rgba: [int(255 * c) for c in rgba[:3]] + [160])
    dataframe.reset_index(inplace=True)
    return dataframe, colname

def select_plotting(data):
    st.write('View results by station for key indicators of urban heat in 2023 in Biel. Select the data type below.')
    datavars = ['4AM UHI', 'Max Temp', 'Min Temp', 'City Index', 'FITNAH Prediction']
    var_selection = st.selectbox('Select variables', datavars)
    plot_points(data, var_selection)

def plot_points(data, selection):
    df, column = prep_data(data, selection)
    # Create a Pydeck layer for colored scatter plot points
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=['X', 'Y'],
        get_color='color',  # Now this refers to the calculated RGBA values
        get_radius=100,
        pickable=True,
        extruded=True,
    )

    # Define view state (center map around mean coordinates)
    view_state = pdk.ViewState(
        latitude=df['Y'].mean(),
        longitude=df['X'].mean(),
        zoom=12,
        pitch=30,
    )

    # Create the map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": f"Sensor: {{sensor}}\nName: {{name}}\nValue: {{{column}}}"},
    )

    # Render the map in Streamlit
    st.pydeck_chart(r)


