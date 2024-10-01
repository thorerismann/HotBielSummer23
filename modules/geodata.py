import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from rasterio.mask import mask
from pathlib import Path
import pandas as pd
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

directories = {
    'plot_directory': Path('/home/tge/master/urbanheat/charts'),
    'data_directory': Path('/home/tge/master/urbanheat/data/'),
    'data_output': Path('/home/tge/master/urbanheat/processed')
}

raster_paths = dict(fitnahtemp=Path('/home/tge/master/urbanheat/data/fitnahdata/reprojected_temp.tif'),
                                 fitnahuhispace=Path('/home/tge/master/urbanheat/data/fitnahdata/winss20n_reproj.tif'),
                                 fitnahuhistreet=Path('/home/tge/master/urbanheat/data/fitnahdata/winsv20n_reproj.tif'),
                                 dem=Path('/home/tge/master/urbanheat/data/dem/dem_merged.tif'),
                                 slope=Path('/home/tge/master/urbanheat/data/dem/slope.tif'),
                                 landuse=Path('/home/tge/master/urbanheat/data/fitnah_landuse/landuse.tif')
                                 )

shape_paths = dict(wind=Path('/home/tge/masterthesis/database/Strömung'),
                   buildings='/home/tge/master/urbanheat/data/buildings/buildings.shp',
                   )

loc_path = Path('/home/tge/master/urbanheat/data/rawbieldata/locations/sensors_locations.csv')

class GeoDataCollector:
    def __init__(self, directories, buffers):

        self.raster_paths = raster_paths
        self.shape_paths = shape_paths
        self.save_path = directories['data_output'] / 'geo_data'
        self.save_path.mkdir(exist_ok=True)
        self.points = self.load_points(loc_path)
        self.buffers = self.create_buffers(buffers, self.points)

    def load_points(self, points_path):
        points = pd.read_csv(points_path)
        points_gpd = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.X, points.Y))
        points_gpd = points_gpd.dropna(subset=['Name']).sort_values(by='Name').set_index('Name', drop=True)
        points_gpd = points_gpd.drop([240, 207], axis=0)
        points_gpd = points_gpd.set_crs('epsg:2056')  # Ensure CRS is set correctly
        return points_gpd

    def create_buffers(self, buffers, points):
        buffer_frame = gpd.GeoDataFrame(index=points.index)
        for buffer in buffers:
            # Assuming buffer sizes are passed as integers representing meters
            buffer_frame[str(buffer)] = points.geometry.buffer(buffer)
        return buffer_frame

    def get_raster_image(self, src, geometry):
        try:
            out_image, out_transform = mask(src, [geometry], crop=True, all_touched=True)
        except:
            print('Error with geometry:', geometry)
        return out_image[0]

    def calculate_statistics(self, data):
        """Calculate statistical metrics from the raster data.
        """
        stats = {
            'mean': np.nan,
            'max': np.nan,
            'min': np.nan,
            'median': np.nan,
            'count': 0,  # Count of non-NaN grid squares
        }

        if data.size > 0:
            stats.update({
                'mean': np.nanmean(data),
                'max': np.nanmax(data),
                'min': np.nanmin(data),
                'median': np.nanmedian(data),
                'count': data.size,
            })

        return stats

    def get_raster_stats(self, path):
        with rasterio.open(path) as src:
            buffered_data = []
            for buffer in self.buffers.columns:  # Iterate over each buffer size
                for idx, row in self.buffers.iterrows():
                    geometry = row[buffer]  # Access the geometry for the current buffer size
                    image = self.get_raster_image(src, geometry)
                    data_clean = image[image != src.nodata]  # Clean the data to ignore nodata
                    stationstats = self.calculate_statistics(data_clean)
                    stationstats['buffer'] = int(buffer)
                    stationstats['logger'] = int(idx)
                    stationstats['geometry'] = geometry
                    buffered_data.append(stationstats)  # Organize results by buffer size

        df = pd.DataFrame(buffered_data)
        return df

    def calculate_landuse_counts(self, data, unique_categories):
        """Calculate counts for each land use category within the data."""
        counts = {int(category): np.sum(data == category) for category in unique_categories}
        return counts

    def get_landuse_stats(self):
        with rasterio.open(self.raster_paths['landuse']) as src:
            # If unique categories are not predefined, you could determine them dynamically:
            entire_image = src.read(1)
            unique_categories = np.unique(entire_image)
            unique_categories = unique_categories[unique_categories != src.nodata]  # Exclude nodata

            buffered_data = []
            for buffer in self.buffers.columns[:-1]:  # Exclude the logger column
                for idx, row in self.buffers.iterrows():
                    geometry = row[buffer]
                    image = self.get_raster_image(src, geometry)
                    land_use_counts = self.calculate_landuse_counts(image, unique_categories)
                    land_use_counts['buffer'] = int(buffer)
                    land_use_counts['logger'] = int(idx)
                    land_use_counts['geometry'] = geometry
                    buffered_data.append(land_use_counts)
        df = pd.DataFrame(buffered_data)
        print(df)
        return df

    def save_buffered_data(self):
        data = self.calculate_rasters()
        Path(self.save_path).mkdir(exist_ok=True)
        path = Path(self.save_path) / 'buffered_data.csv'
        data = data[data['dtype'] != 'landuse']
        data.to_csv(path, index=False)
        data_lu = self.get_landuse_stats()
        path_lu = Path(self.save_path) / 'buffered_landuse.csv'
        data_lu.to_csv(path_lu, index=False)

    def calculate_rasters(self):
        buffered_data = []
        for name, path in self.raster_paths.items():
            data = self.get_raster_stats(path)
            data['dtype'] = name
            buffered_data.append(data)
        final_data = pd.concat(buffered_data)

        return final_data
