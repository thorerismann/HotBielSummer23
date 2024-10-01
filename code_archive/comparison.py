import itertools
import seaborn as sns

from matplotlib import pyplot as plt

from code_archive import charts
import pandas as pd
from scipy.stats import kendalltau, spearmanr
uhi_22, uhi_23, tn_22, tn_23, sd_22, sd_23, sensor_names, meteo22, meteo23, data22, data23, sensor_names = charts.main(plot=False)

buff1 = '/home/tge/masterthesis/app/database/buffered_data.csv'
buff2 = '/home/tge/masterthesis/app/database/buffered_landuse.csv'
df1 = pd.read_csv(buff1)
df2 = pd.read_csv(buff2)
df1 = df1.rename(columns={'logger':'sensor', 'max':'maxtemp', 'min':'mintemp', 'mean':'meantemp', 'median':'mediantemp'})
sensor_names = sensor_names.rename(columns={'Name':'sensor'})
new = pd.merge(df1, sensor_names, on='sensor')
df2 = df2.rename(columns={'logger':'sensor'})
new2 = pd.merge(df2, sensor_names, on='sensor')
new.to_csv('/home/tge/masterthesis/latek/images/results/buffered_data.csv', index=False)
df1.set_index(['sensor', 'buffer', 'dtype'], inplace=True)
ds1 = df1.to_xarray()


new[(new.buffer == 10) & (new.dtype == 'fitnahtemp')].to_csv('/home/tge/masterthesis/latek/images/results/buffered_temp_10.csv', index=False)

new[(new.buffer == 50) & (new.dtype == 'fitnahtemp')].to_csv('/home/tge/masterthesis/latek/images/results/buffered_temp_50.csv', index=False)

new[(new.buffer == 5) & (new.dtype == 'fitnahtemp')].to_csv('/home/tge/masterthesis/latek/images/results/buffered_temp_5.csv', index=False)
def rank_landuse(data):
    land_use_classes = {
        '20': "einzelne, freistehende Gebäude",
        '22': "Strassen",
        '14': "Gewässer",
        '7': "Gleis",
        '9': "Rasen",
        '15': "Rebflächen",
        '24': "versiegelt plus hohe Vegetation",
        '25': "hohe Vegetation",
        '10': "kleinere Wege",
        '16': "freiliegender Fels",
        '27': "naturferner Boden ohne Vegetation"
    }
    data = data.rename(columns=land_use_classes)
    print(data.columns)
    full_ranking = []
    for buffer in data.buffer.unique():
        for dtype in [x for x in data.columns if x not in ['buffer', 'sensor', 'X', 'Y', 'Long_name']]:
            lu = data[(data.buffer == buffer)][['sensor', dtype]].copy()
            lu.sort_values(dtype, inplace=True)
            lu = lu.reset_index(drop=True)
            lu = lu.reset_index(drop=False)
            lu = lu.rename(columns={'index': f'rank_{buffer}_{dtype}'})
            lu[f'rank_{buffer}_{dtype}'] = lu[f'rank_{buffer}_{dtype}'] + 1
            lu.set_index('sensor', inplace=True)
            full_ranking.append(lu[f'rank_{buffer}_{dtype}'])
    return pd.concat(full_ranking, axis=1).sort_index()
ranked_landuse = rank_landuse(new2)
def rank_fitnah_buffer(data, variable):
    print(data.columns)
    full_ranking = []
    for buffer in data.buffer.unique():
        ft = data[(data.buffer == buffer) & (data.dtype == 'fitnahtemp')].copy()
        ft = ft.sort_values(variable)
        ft = ft.reset_index(drop=True)
        ft = ft.reset_index(drop=False)
        ft = ft.rename(columns={'index': f'rank_{buffer}_fitnahtemp'})
        ft[f'rank_{buffer}_fitnahtemp'] = ft[f'rank_{buffer}_fitnahtemp'] + 1
        ft.set_index('sensor', inplace=True)

        fstreet = data[(data.buffer == buffer) & (data.dtype == 'fitnahuhistreet')].copy()
        fstreet = fstreet.sort_values(variable)
        fstreet = fstreet.reset_index(drop=True)
        fstreet = fstreet.reset_index(drop=False)
        fstreet = fstreet.rename(columns={'index': f'rank_{buffer}_fitnahstreet'})
        fstreet[f'rank_{buffer}_fitnahstreet'] = fstreet[f'rank_{buffer}_fitnahstreet'] + 1
        fstreet.set_index('sensor', inplace=True)

        fspace = data[(data.buffer == buffer) & (data.dtype == 'fitnahuhispace')].copy()
        fspace = fspace.sort_values(variable)
        fspace = fspace.reset_index(drop=True)
        fspace = fspace.reset_index(drop=False)
        fspace = fspace.rename(columns={'index': f'rank_{buffer}_fitnahspace'})
        fspace[f'rank_{buffer}_fitnahspace'] = fspace[f'rank_{buffer}_fitnahspace'] + 1
        fspace.set_index('sensor', inplace=True)



        ranked_fitnah = pd.concat([ft[f'rank_{buffer}_fitnahtemp'], fstreet[f'rank_{buffer}_fitnahstreet'], fspace[f'rank_{buffer}_fitnahspace']], axis=1)
        full_ranking.append(ranked_fitnah)
    return pd.concat(full_ranking, axis=1).sort_index()
ranked_fitnah = rank_fitnah_buffer(new, variable='meantemp')
ranked_fitnah.to_csv('/home/tge/masterthesis/latek/images/results/ranked_fitnah.csv', index=False)

times = [['2023-05-15', '2023-09-15'], ['2023-08-14', '2023-08-24']]
def rank_station_data(tn23, uhi23, sd23, data23, times):

    uhi_206 = uhi23.sel(time=slice(times[0][0], times[0][1])).uhi_206.groupby(uhi23.sel(time=slice(times[0][0], times[0][1])).time.dt.hour).mean().to_dataframe().reset_index()
    city_index = uhi23.sel(time=slice(times[0][0], times[0][1])).city_index.groupby(uhi23.sel(time=slice(times[0][0], times[0][1])).time.dt.hour).mean().to_dataframe().reset_index()
    print(uhi_206)
    print(uhi_206.columns)
    fiveamuhi = uhi_206[uhi_206.hour == 5].copy().sort_values('uhi_206').reset_index(drop=True).reset_index(drop=False)
    fouramuhi = uhi_206[uhi_206.hour == 4].copy().sort_values('uhi_206').reset_index(drop=True).reset_index(drop=False)
    fiveamcity = city_index[city_index.hour == 5].copy().sort_values('city_index').reset_index(drop=True).reset_index(drop=False)
    fouramcity = city_index[city_index.hour == 4].copy().sort_values('city_index').reset_index(drop=True).reset_index(drop=False)
    fiveamuhi = fiveamuhi.rename(columns={'index': 'rank_uhi_5am_23'})
    fiveamuhi['rank_uhi_5am_23'] = fiveamuhi['rank_uhi_5am_23'] + 1
    fiveamuhi.set_index('sensor', inplace=True)
    fouramuhi = fouramuhi.rename(columns={'index': 'rank_uhi_4am_23'})
    fouramuhi['rank_uhi_4am_23'] = fouramuhi['rank_uhi_4am_23'] + 1
    fouramuhi.set_index('sensor', inplace=True)
    fiveamcity = fiveamcity.rename(columns={'index': 'rank_city_5am_23'})
    fiveamcity['rank_city_5am_23'] = fiveamcity['rank_city_5am_23'] + 1
    fiveamcity.set_index('sensor', inplace=True)
    fouramcity = fouramcity.rename(columns={'index': 'rank_city_4am_23'})
    fouramcity['rank_city_4am_23'] = fouramcity['rank_city_4am_23'] + 1
    fouramcity.set_index('sensor', inplace=True)

    rank_23 = pd.concat([fiveamuhi['rank_uhi_5am_23'], fouramuhi['rank_uhi_4am_23'], fiveamcity['rank_city_5am_23'], fouramcity['rank_city_4am_23']], axis=1)

    uhi_206 = uhi23.sel(time=slice(times[1][0], times[1][1])).uhi_206.groupby(uhi23.sel(time=slice(times[1][0], times[1][1])).time.dt.hour).mean().to_dataframe().reset_index()
    city_index = uhi23.sel(time=slice(times[1][0], times[1][1])).city_index.groupby(uhi23.sel(time=slice(times[1][0], times[1][1])).time.dt.hour).mean().to_dataframe().reset_index()

    fiveamuhi = uhi_206[uhi_206.hour == 5].copy().sort_values('uhi_206').reset_index(drop=True).reset_index(
        drop=False)
    fouramuhi = uhi_206[uhi_206.hour == 4].copy().sort_values('uhi_206').reset_index(drop=True).reset_index(
        drop=False)
    fiveamcity = city_index[city_index.hour == 5].copy().sort_values('city_index').reset_index(
        drop=True).reset_index(drop=False)
    fouramcity = city_index[city_index.hour == 4].copy().sort_values('city_index').reset_index(
        drop=True).reset_index(drop=False)
    fiveamuhi = fiveamuhi.rename(columns={'index': 'rank_uhi_5am_hw'})
    fiveamuhi['rank_uhi_5am_hw'] = fiveamuhi['rank_uhi_5am_hw'] + 1
    fiveamuhi.set_index('sensor', inplace=True)
    fouramuhi = fouramuhi.rename(columns={'index': 'rank_uhi_4am_hw'})
    fouramuhi['rank_uhi_4am_hw'] = fouramuhi['rank_uhi_4am_hw'] + 1
    fouramuhi.set_index('sensor', inplace=True)
    fiveamcity = fiveamcity.rename(columns={'index': 'rank_city_5am_hw'})
    fiveamcity['rank_city_5am_hw'] = fiveamcity['rank_city_5am_hw'] + 1
    fiveamcity.set_index('sensor', inplace=True)
    fouramcity = fouramcity.rename(columns={'index': 'rank_city_4am_hw'})
    fouramcity['rank_city_4am_hw'] = fouramcity['rank_city_4am_hw'] + 1
    fouramcity.set_index('sensor', inplace=True)

    rank_hw = pd.concat([fiveamuhi['rank_uhi_5am_hw'], fouramuhi['rank_uhi_4am_hw'], fiveamcity['rank_city_5am_hw'], fouramcity['rank_city_4am_hw']], axis=1)

    full_station = pd.concat([rank_23, rank_hw], axis=1)

    return full_station

stationrank = rank_station_data(tn_23, uhi_23, sd_23, data23, times).sort_index()
results = []

# Loop over all combinations of model and empirical data columns
for model_col, emp_col in itertools.product(ranked_fitnah.columns, stationrank.columns):
    # Calculate Kendall's Tau and Spearman's Rho
    kendall_tau, kendall_p_value = kendalltau(ranked_fitnah[model_col], stationrank[emp_col])
    spearman_rho, spearman_p_value = spearmanr(ranked_fitnah[model_col], stationrank[emp_col])

    # Store the results as a dictionary
    results.append({
        'model_column': model_col,
        'empirical_column': emp_col,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p_value,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p_value
    })

# Convert results into a DataFrame for easy viewing
results_df = pd.DataFrame(results)
results_df.to_csv('/home/tge/masterthesis/latek/images/results/spearman.csv')

results = []
# Loop over all combinations of model and empirical data columns
for model_col, emp_col in itertools.product(ranked_landuse.columns, stationrank.columns):
    # Calculate Kendall's Tau and Spearman's Rho
    kendall_tau, kendall_p_value = kendalltau(ranked_landuse[model_col], stationrank[emp_col])
    spearman_rho, spearman_p_value = spearmanr(ranked_landuse[model_col], stationrank[emp_col])

    # Store the results as a dictionary
    results.append({
        'model_column': model_col,
        'empirical_column': emp_col,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p_value,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p_value
    })

# Convert results into a DataFrame for easy viewing
results_df_lu = pd.DataFrame(results)
results_df_lu.to_csv('/home/tge/masterthesis/latek/images/results/spearman_landuse.csv')

def plot_results_all_buffers(data, variable, nicevar):
    sns.set_context("talk", font_scale=1.2)
    ft = data[data.model_column.str.contains('fitnahtemp')].copy()
    buffer = ft.model_column.str.extract(r'(\d+)').astype(int)
    ft['buffer'] = buffer
    # now plot the results
    # x axis is buffer
    # y axis is spearman_rho
    # hue is empirical column
    plt.figure(figsize=(12, 8))

    # Create the plot
    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=ft, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of FITNAH Temperature Data at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Column', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_temp.png')
    plt.show()

    sp = data[data.model_column.str.contains('fitnahspace')].copy()
    buffer = sp.model_column.str.extract(r'(\d+)').astype(int)
    sp['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=sp, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of FITNAH UHI Space Data at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.xscale('log')
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_space.png')
    plt.show()

    st = data[data.model_column.str.contains('fitnahstreet')].copy()
    buffer = st.model_column.str.extract(r'(\d+)').astype(int)
    st['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))

    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=st, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of FITNAH UHI Street Data at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_street.png')
    plt.show()
#
plot_results_all_buffers(results_df, 'spearman_rho', 'Spearman\'s Rho')
plot_results_all_buffers(results_df, 'kendall_tau', 'Kendall\'s Tau')


def plot_results_lu_all_buffers(data, variable, nicevar):
    sns.set_context("talk", font_scale=1.2)
    gleis = data[data.model_column.str.contains('Gleis')].copy()
    buffer = gleis.model_column.str.extract(r'(\d+)').astype(int)
    gleis['buffer'] = buffer
    # now plot the results
    # x axis is buffer
    # y axis is spearman_rho
    # hue is empirical column
    plt.figure(figsize=(12, 8))

    # Create the plot
    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=gleis, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of Gleis (LU 7) Temperature Data at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_gleis.png')
    plt.show()

    rasen = data[data.model_column.str.contains('Rasen')].copy()
    buffer = rasen.model_column.str.extract(r'(\d+)').astype(int)
    rasen['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=rasen, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of Grass  at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.xscale('log')
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_rasen.png')
    plt.show()

    einzelne = data[data.model_column.str.contains('einzelne')].copy()
    buffer = einzelne.model_column.str.extract(r'(\d+)').astype(int)
    einzelne['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))

    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=einzelne, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of Single Buildings at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_einzelne.png')
    plt.show()

    einzelne = data[data.model_column.str.contains('Strassen')].copy()
    buffer = einzelne.model_column.str.extract(r'(\d+)').astype(int)
    einzelne['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))

    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=einzelne, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} of Streets at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_strassen.png')
    plt.show()

    einzelne = data[data.model_column.str.contains('versiegelt')].copy()
    buffer = einzelne.model_column.str.extract(r'(\d+)').astype(int)
    einzelne['buffer'] = buffer
    # now plot the results
    plt.figure(figsize=(12, 8))

    sns.lineplot(x='buffer', y=variable, hue='empirical_column', data=einzelne, marker='o')

    # Add labels and title
    plt.xlabel('Buffer Size')
    plt.xscale('log')
    plt.ylabel(nicevar)
    plt.title(f'{nicevar} Buildings and tall vegetation at varying buffer sizes')

    # Show the plot
    plt.legend(title='Empirical Data', loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/tge/masterthesis/latek/images/results/{variable}_verseigelt.png')
    plt.show()


plot_results_lu_all_buffers(results_df_lu, 'spearman_rho', 'Spearman\'s Rho')
plot_results_lu_all_buffers(results_df_lu, 'kendall_tau', 'Kendall\'s Tau')