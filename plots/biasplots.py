import xarray as xr
from matplotlib import pyplot as plt
import seaborn as sns


def calculate_bias(meteodata, tdata, directories):
    gre = meteodata.temperature.sel(stn='GRE')
    diff206 = tdata.sel(sensor=206) - gre
    diff207 = tdata.sel(sensor=207) - gre
    diff206_207 = tdata.sel(sensor=206) - tdata.sel(sensor=207)
    dsdict = {'diff206': diff206, 'diff207': diff207, 'diff206_207': diff206_207}
    dsdict_squeezed = {k: v.drop_vars(['stn', 'sensor'], errors='ignore') for k, v in dsdict.items()}
    for dataset in dsdict_squeezed.values():
        print(dataset)
    newds = xr.Dataset(dsdict_squeezed)
    directory = directories['data_output'] / 'bias_data'
    directory.mkdir(parents=True, exist_ok=True)
    newds.to_netcdf(directories['data_directory'] / 'bias_data' / 'bias_data.nc')
    return newds

def plot_hourly_line(bdata, directories, title):
    grouped_mean = bdata.groupby(bdata.time.dt.hour).mean().to_dataframe()
    grouped_std = bdata.groupby(bdata.time.dt.hour).std().to_dataframe()

    color_map = ['red', 'purple', 'blue']
    fontsize = 14
    plt.figure(figsize=(12, 6))
    print(grouped_mean)
    for i, column in enumerate(grouped_mean.columns):
        color = color_map[i]
        grouped_mean[column].plot(label=column, color=color)
        plt.fill_between(grouped_mean.index, grouped_mean[column] - grouped_std[column],
                         grouped_mean[column] + grouped_std[column], color=color, alpha=0.2)
    plt.xlabel('Hour of the Day', fontsize=fontsize)
    plt.ylabel('Temperature (°C)', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()

    directory = directories['plot_directory'] / 'biasplots'
    directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(directory / 'hourly_bias_line.png')
    plt.close()

    directory = directories['data_output'] / 'bias_data'
    directory.mkdir(parents=True, exist_ok=True)
    grouped_mean.to_csv(directory / 'hourly_bias_mean.csv')
    grouped_std.to_csv(directory / 'hourly_bias_std.csv')

def plot_hourly_box(bias, directories, title):
    for datavar in bias.data_vars:
        diff_df = bias[datavar].to_dataframe().reset_index()
        diff_df['hour'] = diff_df['time'].dt.hour

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Create box plots for each hour
        sns.boxplot(x='hour', y=datavar, data=diff_df, ax=ax)

        ax.set_xlabel('Hour of the Day', fontsize=14)
        ax.set_ylabel('Bias (°C)', fontsize=14)
        ax.set_title(title + f' for {datavar}', fontsize=16)
        ax.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        save_path = directories['plot_directory'] / 'biasplots' / f'hourly_bias_box_{datavar}.png'
        plt.savefig(save_path)
        plt.close()