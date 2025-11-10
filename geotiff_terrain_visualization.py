#!/usr/bin/env python3
"""
GeoTIFF Terrain Visualization using GDAL
=======================================

This script loads terrain data directly from GeoTIFF using GDAL and creates
3D visualizations. This provides an alternative to the GeoJSON approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from osgeo import gdal
import sys

# PV Array characteristics
LAT_CENTER, LON_CENTER = 39.796678, -79.092463  # Meyersdale, PA
ARRAY_WIDTH_M, ARRAY_HEIGHT_M = 16.8, 4.6  # 55ft x 15ft converted to meters

def load_geotiff_data(geotiff_file, sample_factor=1):
    """
    Load elevation data from GeoTIFF file using GDAL.
    
    Parameters:
    -----------
    geotiff_file : str
        Path to the GeoTIFF file
    sample_factor : int
        Factor to subsample the data (1 = full resolution, 2 = every other pixel, etc.)
    
    Returns:
    --------
    tuple : (x_coords, y_coords, elevations)
        Arrays of x, y coordinates and elevation values
    """
    print(f"Loading GeoTIFF data from {geotiff_file}...")
    
    # Open the GeoTIFF file
    dataset = gdal.Open(geotiff_file)
    if dataset is None:
        raise FileNotFoundError(f"Could not open {geotiff_file}")
    
    # Get geotransform parameters
    geotransform = dataset.GetGeoTransform()
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    print(f"Geotransform: {geotransform}")
    
    # Read the elevation band
    band = dataset.GetRasterBand(1)
    elevation_array = band.ReadAsArray()
    
    # Get dimensions
    rows, cols = elevation_array.shape
    print(f"Original dimensions: {rows} rows × {cols} columns")
    
    # Subsample if requested
    if sample_factor > 1:
        elevation_array = elevation_array[::sample_factor, ::sample_factor]
        rows, cols = elevation_array.shape
        print(f"Subsampled dimensions: {rows} rows × {cols} columns (factor: {sample_factor})")
        pixel_width *= sample_factor
        pixel_height *= sample_factor
    
    # Create coordinate arrays
    x_coords = np.arange(cols) * pixel_width + x_origin
    y_coords = np.arange(rows) * pixel_height + y_origin
    
    # Create meshgrid
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    
    # Flatten for scatter data
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    elev_flat = elevation_array.flatten()
    
    # Remove no-data values
    nodata_value = band.GetNoDataValue()
    if nodata_value is not None:
        valid_mask = elev_flat != nodata_value
        x_flat = x_flat[valid_mask]
        y_flat = y_flat[valid_mask]
        elev_flat = elev_flat[valid_mask]
    
    print(f"Coordinate range: X={x_flat.min():.1f} to {x_flat.max():.1f}")
    print(f"Coordinate range: Y={y_flat.min():.1f} to {y_flat.max():.1f}")
    print(f"Elevation range: {elev_flat.min():.1f}m to {elev_flat.max():.1f}m")
    print(f"Valid data points: {len(elev_flat):,}")
    
    # Clean up
    dataset = None
    
    return x_flat, y_flat, elev_flat, x_mesh, y_mesh, elevation_array

def plot_geotiff_terrain(x_coords, y_coords, elevations, x_mesh, y_mesh, elev_array):
    """Create visualizations from GeoTIFF data."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D surface plot using the gridded data
    ax1 = fig.add_subplot(131, projection='3d')
    surface = ax1.plot_surface(x_mesh, y_mesh, elev_array, cmap='terrain', 
                              alpha=0.8, linewidth=0, antialiased=True)
    
    # Add array location (approximate center)
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2
    center_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                               (x_center, y_center), method='linear')
    
    if np.isnan(center_elevation):
        center_elevation = np.nanmean(elevations)
    
    # Simple array outline
    array_size = 50  # meters for visualization
    array_x = np.array([x_center - array_size/2, x_center + array_size/2, 
                       x_center + array_size/2, x_center - array_size/2, 
                       x_center - array_size/2])
    array_y = np.array([y_center - array_size/2, y_center - array_size/2,
                       y_center + array_size/2, y_center + array_size/2,
                       y_center - array_size/2])
    array_z = np.full_like(array_x, center_elevation)
    
    ax1.plot(array_x, array_y, array_z, 'r-', linewidth=3, label='PV Array')
    ax1.scatter(x_center, y_center, center_elevation, color='red', s=100)
    
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_zlabel('Elevation (m)')
    ax1.set_title('3D Terrain Surface\n(GeoTIFF Data)')
    ax1.legend()
    ax1.view_init(elev=30, azim=45)
    
    # 2D contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(x_mesh, y_mesh, elev_array, levels=20, cmap='terrain')
    contour_lines = ax2.contour(x_mesh, y_mesh, elev_array, levels=10, 
                               colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%d m')
    
    ax2.plot(array_x, array_y, 'r-', linewidth=2, label='PV Array')
    ax2.plot(x_center, y_center, 'ro', markersize=8)
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.set_title('Terrain Contour Map\n(GeoTIFF Data)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    fig.colorbar(contour, ax=ax2, label='Elevation (m)')
    
    # Histogram of elevations
    ax3 = fig.add_subplot(133)
    ax3.hist(elevations, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(center_elevation, color='red', linestyle='--', linewidth=2, 
               label=f'Array elevation: {center_elevation:.1f}m')
    ax3.set_xlabel('Elevation (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Elevation Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nTerrain Statistics:")
    print(f"  Area coverage: {(x_coords.max()-x_coords.min()):.0f}m × {(y_coords.max()-y_coords.min()):.0f}m")
    print(f"  Elevation range: {elevations.min():.1f}m to {elevations.max():.1f}m")
    print(f"  Mean elevation: {elevations.mean():.1f}m")
    print(f"  Standard deviation: {elevations.std():.1f}m")
    print(f"  Array location (approx): X={x_center:.1f}, Y={y_center:.1f}, Z={center_elevation:.1f}m")
    
    return fig

def main():
    """Main function to load and visualize GeoTIFF data."""
    print("🗻 GeoTIFF Terrain Visualization")
    print("=" * 40)
    
    # GeoTIFF file path
    geotiff_file = 'data/USGS_1M_17_x66y441_PA_WesternPA_2019_D20_trimmed_500m.tif'
    
    try:
        # Load data with subsampling for performance
        x_coords, y_coords, elevations, x_mesh, y_mesh, elev_array = load_geotiff_data(
            geotiff_file, sample_factor=2)  # Subsample by factor of 2 for performance
        
        # Create visualization
        print("\nCreating visualizations...")
        fig = plot_geotiff_terrain(x_coords, y_coords, elevations, x_mesh, y_mesh, elev_array)
        
        plt.show()
        
        print("\nVisualization complete!")
        print("Note: The array location shown is approximate center of the terrain data.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the GeoTIFF file exists in the data directory.")
    except Exception as e:
        print(f"Error loading GeoTIFF data: {e}")
        print("This may be due to GDAL installation or file format issues.")

if __name__ == "__main__":
    main()