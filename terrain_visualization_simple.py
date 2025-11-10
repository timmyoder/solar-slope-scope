#!/usr/bin/env python3
"""
3D Terrain Elevation Visualization (Simplified)
==============================================

This script loads USGS elevation data from GeoJSON format and creates a 3D surface plot
to visualize the terrain. This version works with projected coordinates and doesn't
require coordinate system conversion.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def load_elevation_data(geojson_file):
    """Load elevation data from GeoJSON file."""
    print("Loading elevation data from GeoJSON...")
    
    with open(geojson_file, 'r') as f:
        data = json.load(f)
    
    x_coords, y_coords, elevations = [], [], []
    
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        
        # Extract coordinates (projected coordinate system)
        x, y = coords[0], coords[1]
        elevation = props.get('elevation')
            
        if elevation is not None:
            x_coords.append(x)
            y_coords.append(y)
            elevations.append(elevation)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    elevations = np.array(elevations)
    
    print(f"Loaded {len(elevations)} elevation points")
    print(f"X range: {x_coords.min():.1f} to {x_coords.max():.1f}")
    print(f"Y range: {y_coords.min():.1f} to {y_coords.max():.1f}")
    print(f"Elevation range: {elevations.min():.1f}m to {elevations.max():.1f}m")
    
    return x_coords, y_coords, elevations

def create_terrain_surface(x_coords, y_coords, elevations, resolution=100):
    """Create a regular grid surface from scattered elevation points."""
    print("Creating terrain surface mesh...")
    
    # Create regular grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    # Interpolate elevation data onto regular grid
    points = np.column_stack((x_coords, y_coords))
    elev_mesh = griddata(points, elevations, (x_mesh, y_mesh), method='linear')
    
    # Fill any NaN values with nearest neighbor interpolation
    mask = np.isnan(elev_mesh)
    if np.any(mask):
        print("Filling NaN values with nearest neighbor interpolation...")
        elev_mesh_nearest = griddata(points, elevations, (x_mesh, y_mesh), method='nearest')
        elev_mesh[mask] = elev_mesh_nearest[mask]
    
    return x_mesh, y_mesh, elev_mesh

def create_simple_array_outline(x_center, y_center, width_m=16.8, height_m=4.6):
    """Create a simple rectangular outline for the PV array in projected coordinates."""
    half_width = width_m / 2
    half_height = height_m / 2
    
    # Array corners
    x_corners = np.array([
        x_center - half_width,  # SW corner
        x_center + half_width,  # SE corner
        x_center + half_width,  # NE corner
        x_center - half_width,  # NW corner
        x_center - half_width   # Close the polygon
    ])
    
    y_corners = np.array([
        y_center - half_height,  # SW corner
        y_center - half_height,  # SE corner
        y_center + half_height,  # NE corner
        y_center + half_height,  # NW corner
        y_center - half_height   # Close the polygon
    ])
    
    return x_corners, y_corners

def plot_3d_terrain(x_mesh, y_mesh, elev_mesh, array_x=None, array_y=None, array_elev=None):
    """Create 3D surface plot of terrain."""
    print("Creating 3D terrain visualization...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot terrain surface
    surface = ax.plot_surface(x_mesh, y_mesh, elev_mesh, 
                             cmap='terrain', alpha=0.8, linewidth=0, antialiased=True)
    
    # Plot array location if provided
    if array_x is not None and array_y is not None and array_elev is not None:
        ax.plot(array_x, array_y, array_elev, 'r-', linewidth=3, label='PV Array Outline')
        ax.scatter(array_x[0], array_y[0], array_elev[0], color='red', s=100, label='Array Location')
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=20, label='Elevation (m)')
    
    # Labels and title
    ax.set_xlabel('X Coordinate (projected)')
    ax.set_ylabel('Y Coordinate (projected)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Terrain Visualization\nUSGS Elevation Data', fontsize=14)
    
    if array_x is not None:
        ax.legend()
    
    # Improve viewing angle
    ax.view_init(elev=30, azim=45)
    
    return fig, ax

def plot_2d_contour(x_mesh, y_mesh, elev_mesh, array_x=None, array_y=None):
    """Create 2D contour plot of terrain."""
    print("Creating 2D contour visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create contour plot
    contour = ax.contourf(x_mesh, y_mesh, elev_mesh, levels=20, cmap='terrain')
    contour_lines = ax.contour(x_mesh, y_mesh, elev_mesh, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%d m')
    
    # Plot array location if provided
    if array_x is not None and array_y is not None:
        ax.plot(array_x, array_y, 'r-', linewidth=2, label='PV Array Outline')
        ax.plot(array_x[0], array_y[0], 'ro', markersize=8, label='Array Location')
        ax.legend()
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Elevation (m)', rotation=270, labelpad=15)
    
    # Labels and formatting
    ax.set_xlabel('X Coordinate (projected)')
    ax.set_ylabel('Y Coordinate (projected)')
    ax.set_title('Terrain Contour Map\nUSGS Elevation Data', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax

def main():
    """Main function to load data and create visualizations."""
    print("🌄 Terrain Elevation Visualization")
    print("=" * 50)
    
    # Load elevation data
    geojson_file = 'data/terrain_elevation_points.geojson'
    
    try:
        x_coords, y_coords, elevations = load_elevation_data(geojson_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {geojson_file}")
        print("Please ensure the GeoJSON file exists in the data directory.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create terrain surface
    x_mesh, y_mesh, elev_mesh = create_terrain_surface(x_coords, y_coords, elevations, resolution=150)
    
    # Create a simple array outline in the center of the data for reference
    # (This is just for visualization - you'd need coordinate conversion for exact placement)
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2
    
    # Get elevation at center
    center_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations, 
                               (x_center, y_center), method='linear')
    if np.isnan(center_elevation):
        center_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations, 
                                   (x_center, y_center), method='nearest')
    
    print(f"Data center location: X={x_center:.1f}, Y={y_center:.1f}")
    print(f"Center elevation: {center_elevation:.1f}m")
    
    # Create array outline (55ft x 15ft = ~16.8m x 4.6m)
    array_x, array_y = create_simple_array_outline(x_center, y_center)
    array_elevations = np.full_like(array_x, center_elevation)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 3D plot
    fig_3d, ax_3d = plot_3d_terrain(x_mesh, y_mesh, elev_mesh, 
                                    array_x, array_y, array_elevations)
    
    # 2D contour plot
    fig_2d, ax_2d = plot_2d_contour(x_mesh, y_mesh, elev_mesh, 
                                    array_x, array_y)
    
    # Display statistics
    print("\n📊 Terrain Statistics:")
    print(f"   Elevation range: {elevations.min():.1f}m to {elevations.max():.1f}m")
    print(f"   Elevation std dev: {elevations.std():.1f}m")
    print(f"   Data coverage: {len(elevations):,} points")
    print(f"   Area coverage: {(x_coords.max()-x_coords.min()):.0f}m × {(y_coords.max()-y_coords.min()):.0f}m")
    
    plt.tight_layout()
    plt.show()
    
    print("\n✨ Visualization complete!")
    print("🔍 Review the plots to verify terrain data matches expected topography")
    print("📝 Note: Array location is approximate - use coordinate conversion for precise placement")

if __name__ == "__main__":
    main()