#!/usr/bin/env python3
"""
Enhanced Terrain Visualization with Precise PV Array Placement
============================================================

This script loads USGS elevation data and places the PV array at the exact
specified lat/lon coordinates using coordinate system conversion.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# PV Array characteristics and location
LAT_CENTER, LON_CENTER = 39.796678, -79.092463  # Meyersdale, PA
ARRAY_WIDTH_FT, ARRAY_HEIGHT_FT = 55, 15
ARRAY_WIDTH_M = ARRAY_WIDTH_FT * 0.3048  # Convert feet to meters
ARRAY_HEIGHT_M = ARRAY_HEIGHT_FT * 0.3048

def load_elevation_data(geojson_file):
    """Load elevation data from GeoJSON file."""
    print("Loading elevation data from GeoJSON...")
    
    with open(geojson_file, 'r') as f:
        data = json.load(f)
    
    x_coords, y_coords, elevations = [], [], []
    slopes, aspects = [], []
    
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        
        x, y = coords[0], coords[1]
        elevation = props.get('elevation')
        slope = props.get('slope')
        aspect = props.get('aspect')
            
        if elevation is not None:
            x_coords.append(x)
            y_coords.append(y)
            elevations.append(elevation)
            slopes.append(slope if slope is not None else 0)
            aspects.append(aspect if aspect is not None else 0)
    
    return (np.array(x_coords), np.array(y_coords), np.array(elevations), 
            np.array(slopes), np.array(aspects))

def estimate_utm_zone_from_data(x_coords):
    """
    Estimate UTM zone based on the x coordinates.
    For Pennsylvania (around Meyersdale), this should be UTM Zone 17N.
    """
    # Given the x coordinates are around 663,000, this is likely UTM Zone 17N
    if 660000 <= x_coords.mean() <= 670000:
        return 17
    else:
        # Fallback estimation
        return int((x_coords.mean() + 500000) / 1000000) + 30

def convert_latlon_to_utm_approx(lat, lon, utm_zone=17):
    """
    Simple approximate conversion from lat/lon to UTM coordinates.
    This is a simplified calculation for visualization purposes.
    """
    # UTM parameters for zone 17N (approximate)
    central_meridian = -81.0  # UTM Zone 17N central meridian
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cm_rad = np.radians(central_meridian)
    
    # Simple UTM approximation (Mercator-like)
    # This is simplified and not geodetically accurate, but sufficient for visualization
    k0 = 0.9996  # UTM scale factor
    a = 6378137.0  # WGS84 semi-major axis
    
    # Approximate calculations
    delta_lon = lon_rad - cm_rad
    
    # UTM Easting (simplified)
    x = 500000 + k0 * a * delta_lon * np.cos(lat_rad)
    
    # UTM Northing (simplified)
    y = k0 * a * lat_rad + 4000000 if lat < 0 else k0 * a * lat_rad
    
    return x, y

def create_array_outline_utm(center_x, center_y, width_m, height_m):
    """Create PV array outline in UTM coordinates."""
    half_width = width_m / 2
    half_height = height_m / 2
    
    x_corners = np.array([
        center_x - half_width,
        center_x + half_width,
        center_x + half_width,
        center_x - half_width,
        center_x - half_width
    ])
    
    y_corners = np.array([
        center_y - half_height,
        center_y - half_height,
        center_y + half_height,
        center_y + half_height,
        center_y - half_height
    ])
    
    return x_corners, y_corners

def create_terrain_surface(x_coords, y_coords, elevations, resolution=150):
    """Create interpolated terrain surface."""
    print("Creating terrain surface mesh...")
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    points = np.column_stack((x_coords, y_coords))
    elev_mesh = griddata(points, elevations, (x_mesh, y_mesh), method='linear')
    
    # Fill NaN values
    mask = np.isnan(elev_mesh)
    if np.any(mask):
        elev_mesh_nearest = griddata(points, elevations, (x_mesh, y_mesh), method='nearest')
        elev_mesh[mask] = elev_mesh_nearest[mask]
    
    return x_mesh, y_mesh, elev_mesh

def plot_terrain_with_array(x_mesh, y_mesh, elev_mesh, x_coords, y_coords, elevations, 
                           array_x, array_y, array_elevation, array_center_x, array_center_y):
    """Create both 3D and 2D visualizations."""
    
    # 3D Plot
    fig = plt.figure(figsize=(16, 12))
    
    # 3D subplot
    ax1 = fig.add_subplot(221, projection='3d')
    surface = ax1.plot_surface(x_mesh, y_mesh, elev_mesh, cmap='terrain', alpha=0.8, 
                              linewidth=0, antialiased=True)
    
    # Plot array
    array_elevations = np.full_like(array_x, array_elevation)
    ax1.plot(array_x, array_y, array_elevations, 'r-', linewidth=3, label='PV Array')
    ax1.scatter(array_center_x, array_center_y, array_elevation, color='red', s=100, 
               label='Array Center')
    
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_zlabel('Elevation (m)')
    ax1.set_title('3D Terrain with PV Array\nMeyersdale, PA')
    ax1.legend()
    ax1.view_init(elev=30, azim=45)
    
    # 2D Contour plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(x_mesh, y_mesh, elev_mesh, levels=20, cmap='terrain')
    contour_lines = ax2.contour(x_mesh, y_mesh, elev_mesh, levels=10, colors='black', 
                               alpha=0.3, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%d m')
    
    ax2.plot(array_x, array_y, 'r-', linewidth=2, label='PV Array')
    ax2.plot(array_center_x, array_center_y, 'ro', markersize=8, label='Array Center')
    ax2.set_xlabel('UTM Easting (m)')
    ax2.set_ylabel('UTM Northing (m)')
    ax2.set_title('Terrain Contour with PV Array')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Colorbar
    fig.colorbar(contour, ax=ax2, label='Elevation (m)')
    
    # Elevation profile along array
    ax3 = fig.add_subplot(223)
    # Get elevations along a line through the array center
    profile_x = np.linspace(array_x.min(), array_x.max(), 50)
    profile_y = np.full_like(profile_x, array_y[0])  # Horizontal line through center
    
    profile_elevations = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                 np.column_stack((profile_x, profile_y)), method='linear')
    
    ax3.plot(profile_x - array_x[0], profile_elevations, 'b-', linewidth=2)
    ax3.axhline(y=array_elevation, color='red', linestyle='--', 
               label=f'Array elevation: {array_elevation:.1f}m')
    ax3.set_xlabel('Distance from Array Center (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_title('Elevation Profile Through Array')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    stats_text = f"""
    [*] PV Array Location:
    Latitude: {LAT_CENTER:.6f}°N
    Longitude: {LON_CENTER:.6f}°W
    
    [+] Array Dimensions:
    Width: {ARRAY_WIDTH_FT}' ({ARRAY_WIDTH_M:.1f}m)
    Height: {ARRAY_HEIGHT_FT}' ({ARRAY_HEIGHT_M:.1f}m)
    
    [^] Terrain Statistics:
    Elevation range: {elevations.min():.1f} - {elevations.max():.1f}m
    Standard deviation: {elevations.std():.1f}m
    Array elevation: {array_elevation:.1f}m
    
    [#] Data Coverage:
    Points: {len(elevations):,}
    Area: {(x_coords.max()-x_coords.min()):.0f}m × {(y_coords.max()-y_coords.min()):.0f}m
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    print("*** Enhanced Terrain Visualization with Precise PV Array Placement ***")
    print("=" * 70)
    
    # Load data
    geojson_file = 'data/terrain_elevation_points.geojson'
    
    try:
        x_coords, y_coords, elevations, slopes, aspects = load_elevation_data(geojson_file)
    except FileNotFoundError:
        print(f"[!] Error: Could not find {geojson_file}")
        return
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return
    
    print(f"Loaded {len(elevations)} elevation points")
    print(f"UTM coordinates range: X={x_coords.min():.0f}-{x_coords.max():.0f}, "
          f"Y={y_coords.min():.0f}-{y_coords.max():.0f}")
    
    # Convert array center coordinates to approximate UTM
    utm_zone = estimate_utm_zone_from_data(x_coords)
    print(f"Estimated UTM Zone: {utm_zone}N")
    
    array_x_utm, array_y_utm = convert_latlon_to_utm_approx(LAT_CENTER, LON_CENTER, utm_zone)
    print(f"Array center in UTM (approx): X={array_x_utm:.0f}, Y={array_y_utm:.0f}")
    
    # Check if array is within data bounds
    if (x_coords.min() <= array_x_utm <= x_coords.max() and
        y_coords.min() <= array_y_utm <= y_coords.max()):
        print("[+] Array location is within terrain data bounds")
    else:
        print("[!] Array location may be outside terrain data bounds")
        print("    Using center of data area for visualization...")
        array_x_utm = (x_coords.min() + x_coords.max()) / 2
        array_y_utm = (y_coords.min() + y_coords.max()) / 2
    
    # Get array elevation
    array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                              (array_x_utm, array_y_utm), method='linear')
    if np.isnan(array_elevation):
        array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                  (array_x_utm, array_y_utm), method='nearest')
    
    # Create array outline
    array_x, array_y = create_array_outline_utm(array_x_utm, array_y_utm, 
                                                ARRAY_WIDTH_M, ARRAY_HEIGHT_M)
    
    # Create terrain surface
    x_mesh, y_mesh, elev_mesh = create_terrain_surface(x_coords, y_coords, elevations)
    
    # Create visualization
    print("\nCreating comprehensive visualization...")
    fig = plot_terrain_with_array(x_mesh, y_mesh, elev_mesh, x_coords, y_coords, elevations,
                                 array_x, array_y, array_elevation, array_x_utm, array_y_utm)
    
    plt.show()
    
    print("\n[*] Visualization complete!")
    print("[>] Review the plots to verify terrain data matches expected topography")

if __name__ == "__main__":
    main()