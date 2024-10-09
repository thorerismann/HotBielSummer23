from matplotlib import pyplot as plt

def overlay_max_mean_min_temperature(mdata, sdata, time_period, title, directories):
    # Define color mapping for each station
    color_map = {'CRM': 'red', 'BER': 'blue', 'GRE': 'green'}
    fontsize = 14
    # Resample the data to daily maximum, mean, and minimum temperatures
    max_temp = mdata.resample(time='1D').max()
    mean_temp = mdata.resample(time='1D').mean()
    min_temp = mdata.resample(time='1D').min()
    
    gre_mean = sdata.sel(sensor=207).resample(time='1D').mean()
    gre_max = sdata.sel(sensor=207).resample(time='1D').max()
    gre_min = sdata.sel(sensor=207).resample(time='1D').min()

    plt.figure(figsize=(12, 6))

    # Loop over each station
    for stn in mdata.stn.values:
        # Ensure the station has a defined color
        color = color_map.get(stn, 'black')

        # Plot maximum temperature
        max_temp.sel(stn=stn)['temperature'].plot(label=f'{stn} Max Temp', linestyle='-', linewidth=2, color=color)

        # Plot mean temperature
        mean_temp.sel(stn=stn)['temperature'].plot(label=f'{stn} Mean Temp', linestyle='--', linewidth=2, color=color)

        # Plot minimum temperature
        min_temp.sel(stn=stn)['temperature'].plot(label=f'{stn} Min Temp', linestyle='-.', linewidth=2, color=color)
    gre_mean.plot(label='207 Mean Temp', linestyle='--', linewidth=2, color='black')
    gre_max.plot(label='207 Max Temp', linestyle='-', linewidth=2, color='black')
    gre_min.plot(label='207 Min Temp', linestyle='-.', linewidth=2, color='black')
    plt.title(title, fontsize=fontsize)
    plt.xlabel('Date', fontsize=fontsize)
    plt.ylabel('Temperature (°C)', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    save_path = directories['plot_directory'] / 'meteoplots' / f'temperature_overlay_{time_period}.png'
    # Save the figure if a save path is provided
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()




def mean_windspeed(data, title, time_period, directories):
    color_map = {'CRM': 'red', 'BER': 'blue', 'GRE': 'green'}
    fontsize = 14

    # Resample the data to daily maximum, mean, and minimum temperatures
    mean_wind= data.resample(time='1D').mean()

    plt.figure(figsize=(12, 6))

    # Loop over each station
    for stn in data.stn.values:
        # Ensure the station has a defined color
        color = color_map.get(stn, 'black')
        mean_wind.sel(stn=stn).windspeed.plot(label=f'{stn} Mean Wind', linewidth=2, color=color)

    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel('Date', fontsize=fontsize)
    plt.ylabel('Windspeed (meters / second)', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if a save path is provided
    save_path = directories['plot_directory'] / 'meteoplots' / f'windspeed_{time_period}.png'
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

def print_max_min(data):
    max_temp = data.max(dim='time')
    min_temp = data.min(dim='time')
