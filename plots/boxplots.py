import matplotlib.pyplot as plt
import seaborn as sns


def plot_hourly_boxplots(data, variable, time_name, time, title, y_label, fontsize, save_dir):
    """
    Function to plot hourly boxplots for UHI 206, UHI 207 and City Index.

    :param data: xarray dataset
    :param variable: variable to plot
    :param time_name: name of the time range
    :param time: time range
    :param title: title of the plot
    :param y_label: y-axis label
    :param fontsize: fontsize
    :param save_dir: directory to save the plots
    :return: None
    """


    # Select the time range
    cut_data = data.sel(time=slice(time[0], time[1]))

    # Extract hour from the time for use in boxplot
    cut_data['hour'] = cut_data.time.dt.hour
    cut_data['doy'] = cut_data.time.dt.dayofyear
    color = "#3498db"

    for station in data.sensor.values:
        station_data = cut_data.sel(sensor=station)

        station_data_daily_hourly_means = station_data.to_dataframe().reset_index().groupby(['doy', 'hour']).mean()
        mydata = station_data_daily_hourly_means.dropna(how='any')
        if len(mydata) < 1:
            continue
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='hour', y=variable, data=mydata, color=color)

        plt.title(f'{title} station {station}', fontsize=fontsize + 2)
        plt.xlabel('Hour of the Day', fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.xticks(range(0, 24))  # Ensure all 24 hours are shown

        plt.tight_layout()
        plt.savefig(save_dir / f'{variable}_{time_name}_{station}.png')
        plt.close()
    mean_data = cut_data.mean(dim='sensor').to_dataframe().reset_index().groupby(['doy', 'hour']).mean()
    print(mean_data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hour', y=variable, data=mean_data, color=color)
    plt.title(f'{title} Mean', fontsize=fontsize + 2)
    plt.xlabel('Hour of the Day', fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(range(0, 24))  # Ensure all 24 hours are shown
    plt.tight_layout()
    plt.savefig(save_dir / f'{variable}_{time_name}_mean.png')
    plt.close()



def plot_hourly_boxplots_temperature(data, time, time_name, title, y_label, fontsize, save_dir):
    # Select the time range
    cut_data = data.sel(time=slice(time[0], time[1]))

    # Extract hour from the time for use in boxplot
    cut_data['hour'] = cut_data.time.dt.hour
    cut_data['doy'] = cut_data.time.dt.dayofyear
    color = "#3498db"

    for station in data.sensor.values:
        station_data = cut_data.sel(sensor=station)

        station_data_daily_hourly_means = station_data.to_dataframe().reset_index().groupby(['doy', 'hour']).mean()
        mydata = station_data_daily_hourly_means.dropna(how='any')
        if len(mydata) < 1:
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='hour', y='temperature', data=mydata, color=color)

        plt.title(f'{title} station {station}', fontsize=fontsize + 2)
        plt.xlabel('Hour of the Day', fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.xticks(range(0, 24))  # Ensure all 24 hours are shown

        plt.tight_layout()
        plt.savefig(save_dir / f'temperature_{time_name}_{station}.png')
        plt.close()
