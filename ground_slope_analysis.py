#!/usr/bin/env python3
"""
Ground Slope Analysis for PV Array Optimization
==============================================

This module provides functions to:
1. Calculate ground slope at a specific location in a given direction
2. Calculate effective array geometry considering ground slope and array tilt
3. Support azimuth optimization for sloped terrain

Key concepts:
- Directional slope: slope of terrain in a specific azimuth direction
- Effective array geometry: combined effect of ground slope + array tilt
- Coordinate systems: lat/lon input, UTM for calculations
"""

import numpy as np
import json
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.spatial.distance import cdist
import warnings

# Try to import coordinate transformation tools
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("⚠️ PyProj not available - using approximate coordinate conversions")

# Constants from project context
LAT_CENTER, LON_CENTER = 39.796678, -79.092463  # Meyersdale, PA
ARRAY_WIDTH_FT, ARRAY_HEIGHT_FT = 55, 15
ARRAY_WIDTH_M = ARRAY_WIDTH_FT * 0.3048
ARRAY_HEIGHT_M = ARRAY_HEIGHT_FT * 0.3048


class TerrainAnalyzer:
    """
    Analyzes terrain data to calculate slopes and gradients for PV array optimization.
    
    Uses GeoJSON point-based terrain data converted from USGS raster grid.
    Grid structure is preserved for efficient nearest-neighbor and gradient calculations.
    """
    
    def __init__(self, terrain_data_file):
        """
        Initialize terrain analyzer.
        
        Parameters:
        -----------
        terrain_data_file : str
            Path to GeoJSON terrain data file (converted from USGS raster)
        """
        self.terrain_file = terrain_data_file
        self._load_geojson_data()
        self._setup_grid_structure()
        
    def _load_geojson_data(self):
        """Load terrain data from GeoJSON format, preserving grid structure."""
        import json
        
        print(f"Loading GeoJSON data from {self.terrain_file}...")
        
        with open(self.terrain_file, 'r') as f:
            data = json.load(f)
        
        # Store all data points with grid info
        self.grid_points = []
        
        for feature in data['features']:
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            
            self.grid_points.append({
                'x': coords[0],
                'y': coords[1], 
                'elevation': props['elevation'],
                'row': props.get('row', -1),
                'col': props.get('col', -1),
                'slope': props.get('slope', 0),
                'aspect': props.get('aspect', 0)
            })
        
        print(f"✅ Loaded {len(self.grid_points)} grid points")
        
        # Extract coordinate arrays for bounds checking
        self.x_coords = np.array([pt['x'] for pt in self.grid_points])
        self.y_coords = np.array([pt['y'] for pt in self.grid_points])
        self.elevations = np.array([pt['elevation'] for pt in self.grid_points])
        
        print(f"   X range: {self.x_coords.min():.1f} - {self.x_coords.max():.1f}")
        print(f"   Y range: {self.y_coords.min():.1f} - {self.y_coords.max():.1f}")
        print(f"   Elevation: {self.elevations.min():.1f} - {self.elevations.max():.1f}m")
    

    
    def _setup_grid_structure(self):
        """Analyze grid structure and setup efficient lookup."""
        print("🔧 Analyzing grid structure...")
        
        # Determine grid spacing by finding closest points
        x_unique = sorted(set(pt['x'] for pt in self.grid_points))
        y_unique = sorted(set(pt['y'] for pt in self.grid_points))
        
        if len(x_unique) > 1:
            self.grid_spacing_x = x_unique[1] - x_unique[0]
        else:
            self.grid_spacing_x = 1.0
            
        if len(y_unique) > 1:
            self.grid_spacing_y = y_unique[1] - y_unique[0] 
        else:
            self.grid_spacing_y = 1.0
            
        print(f"📐 Grid spacing: {self.grid_spacing_x:.1f}m × {self.grid_spacing_y:.1f}m")
        
        # Create bounds for analysis
        self.bounds = {
            'x_min': self.x_coords.min(),
            'x_max': self.x_coords.max(),
            'y_min': self.y_coords.min(), 
            'y_max': self.y_coords.max()
        }
        
        # Create spatial lookup for efficiency
        from scipy.spatial import cKDTree
        points = np.column_stack((self.x_coords, self.y_coords))
        self.kdtree = cKDTree(points)
        
        print("✅ Grid structure analysis complete")
    
    def get_elevation_at_point(self, x, y):
        """
        Get elevation at a specific point using grid-based interpolation.
        
        Parameters:
        -----------
        x, y : float
            UTM coordinates (or projected coordinates matching terrain data)
            
        Returns:
        --------
        float : elevation in meters
        """
        if not (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                self.bounds['y_min'] <= y <= self.bounds['y_max']):
            warnings.warn(f"Point ({x:.0f}, {y:.0f}) outside terrain bounds")
            return np.nan
        
        # Find 4 nearest grid points for bilinear interpolation
        distances, indices = self.kdtree.query([x, y], k=4)
        
        if distances[0] == 0:  # Exact match
            return self.grid_points[indices[0]]['elevation']
        
        # Use inverse distance weighting for sub-grid interpolation
        weights = 1.0 / (distances + 1e-10)  # Avoid division by zero
        weights /= weights.sum()
        
        elevation = sum(weights[i] * self.grid_points[indices[i]]['elevation'] 
                       for i in range(4))
        
        return elevation
    
    def get_slope_in_direction(self, x, y, direction_azimuth):
        """
        Calculate ground slope at a specific point in a given direction.
        
        This is the key function that calculates the directional derivative
        of the terrain surface.
        
        Parameters:
        -----------
        x, y : float
            UTM coordinates
        direction_azimuth : float
            Direction in degrees (0° = North, 90° = East, 180° = South, 270° = West)
            
        Returns:
        --------
        dict : {
            'slope_degrees': float,     # Slope in specified direction (degrees)
            'slope_percent': float,     # Slope as percentage
            'elevation': float,         # Elevation at point
            'gradient_magnitude': float, # Overall terrain gradient magnitude
            'gradient_direction': float  # Direction of steepest slope (degrees)
        }
        """
        if not (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                self.bounds['y_min'] <= y <= self.bounds['y_max']):
            return {
                'slope_degrees': np.nan,
                'slope_percent': np.nan,
                'elevation': np.nan,
                'gradient_magnitude': np.nan,
                'gradient_direction': np.nan
            }
        
        # Find nearest grid point and use pre-computed slope/aspect if available
        distance, nearest_idx = self.kdtree.query([x, y], k=1)
        nearest_point = self.grid_points[nearest_idx]
        
        # If we have pre-computed slope/aspect, use trigonometric projection
        if 'slope' in nearest_point and 'aspect' in nearest_point and nearest_point['slope'] > 0:
            grid_slope = nearest_point['slope']  # Max slope (steepest direction)
            grid_aspect = nearest_point['aspect']  # Direction of max slope
            
            # Calculate angle difference between query direction and steepest slope direction
            angle_diff = abs(direction_azimuth - grid_aspect)
            # Handle wrap-around (e.g., 350° vs 10°)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Use trigonometric projection: slope_in_direction = max_slope * cos(angle_diff)
            # This is accurate for most terrain since slopes follow cosine relationship
            slope_in_direction = grid_slope * np.cos(np.radians(angle_diff))
            
            return {
                'slope_degrees': slope_in_direction,
                'slope_percent': np.tan(np.radians(slope_in_direction)) * 100,
                'elevation': nearest_point['elevation'],
                'gradient_magnitude': np.tan(np.radians(grid_slope)),
                'gradient_direction': grid_aspect,
                'method': f'USGS_trigonometric (angle_diff: {angle_diff:.1f}°)'
            }
        
        # Calculate gradient using finite differences with grid spacing
        direction_rad = np.radians(direction_azimuth)
        
        # Use grid spacing for finite difference calculation
        dx = self.grid_spacing_x * np.cos(np.radians(90 - direction_azimuth))  # East component
        dy = self.grid_spacing_y * np.sin(np.radians(90 - direction_azimuth))  # North component
        
        # Get elevations at points
        elev_center = self.get_elevation_at_point(x, y)
        elev_offset = self.get_elevation_at_point(x + dx, y + dy)
        
        # Calculate slope 
        distance = np.sqrt(dx**2 + dy**2)
        rise = elev_offset - elev_center
        slope_radians = np.arctan(rise / distance)
        slope_degrees = np.degrees(slope_radians)
        slope_percent = np.tan(slope_radians) * 100
        
        # For overall gradient, use nearest point's values or calculate
        if 'slope' in nearest_point and nearest_point['slope'] > 0:
            gradient_magnitude = np.tan(np.radians(nearest_point['slope']))
            gradient_direction = nearest_point['aspect']
        else:
            gradient_magnitude = abs(rise / distance)
            gradient_direction = direction_azimuth  # Approximate
        
        return {
            'slope_degrees': slope_degrees,
            'slope_percent': slope_percent, 
            'elevation': elev_center,
            'gradient_magnitude': gradient_magnitude,
            'gradient_direction': gradient_direction,
            'method': 'finite_difference'
        }
    
    def convert_latlon_to_utm(self, lat, lon):
        """Convert lat/lon to UTM coordinates matching terrain data."""
        if PYPROJ_AVAILABLE:
            try:
                transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32617', always_xy=True)
                x, y = transformer.transform(lon, lat)
                return x, y
            except Exception as e:
                print(f"⚠️ PyProj conversion failed: {e}, using approximate method")
        
        # Fallback approximate conversion for UTM Zone 17N
        central_meridian = -81.0
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        cm_rad = np.radians(central_meridian)
        
        k0 = 0.9996
        a = 6378137.0
        delta_lon = lon_rad - cm_rad
        
        x = 500000 + k0 * a * delta_lon * np.cos(lat_rad)
        y = k0 * a * lat_rad
        
        return x, y
    
    def convert_utm_to_latlon(self, x, y):
        """Convert UTM coordinates back to lat/lon."""
        if PYPROJ_AVAILABLE:
            try:
                transformer = Transformer.from_crs('EPSG:32617', 'EPSG:4326', always_xy=True)
                lon, lat = transformer.transform(x, y)
                return lat, lon
            except Exception as e:
                print(f"⚠️ PyProj conversion failed: {e}, using approximate method")
        
        # Fallback approximate conversion from UTM Zone 17N
        central_meridian = -81.0
        k0 = 0.9996
        a = 6378137.0
        
        # Simple inverse approximation (good enough for small areas)
        lat_rad = y / (k0 * a)
        lon_rad = (x - 500000) / (k0 * a * np.cos(lat_rad)) + np.radians(central_meridian)
        
        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)
        
        return lat, lon


def calculate_effective_array_geometry(ground_slope_degrees, ground_slope_direction,
                                     array_tilt_degrees, array_azimuth_degrees):
    """
    Calculate the effective geometry of a PV array considering ground slope.
    
    This function determines the actual orientation of the PV array surface
    when installed on sloped ground.
    
    Parameters:
    -----------
    ground_slope_degrees : float
        Magnitude of ground slope in the array's facing direction (degrees)
    ground_slope_direction : float  
        Direction of ground slope (degrees, 0° = North)
    array_tilt_degrees : float
        Tilt angle of array relative to ground (degrees)
    array_azimuth_degrees : float
        Azimuth direction array is facing (degrees, 0° = North)
        
    Returns:
    --------
    dict : {
        'effective_tilt': float,        # Actual tilt of array surface (degrees)
        'effective_azimuth': float,     # Effective azimuth accounting for slope
        'ground_slope_component': float, # Component of ground slope in array direction
        'slope_correction': float       # Correction factor applied
    }
    
    Notes:
    ------
    This uses 3D vector geometry to combine:
    1. Ground slope vector (magnitude and direction)
    2. Array tilt vector (relative to ground)
    3. Resulting combined surface normal vector
    """
    
    # Convert angles to radians
    ground_slope_rad = np.radians(ground_slope_degrees)
    ground_dir_rad = np.radians(ground_slope_direction)
    array_tilt_rad = np.radians(array_tilt_degrees)
    array_azimuth_rad = np.radians(array_azimuth_degrees)
    
    # Step 1: Calculate ground slope component in array facing direction
    # This is the directional derivative concept
    direction_diff = ground_dir_rad - array_azimuth_rad
    ground_slope_component = ground_slope_degrees * np.cos(direction_diff)
    
    # Step 2: Calculate 3D surface normal vectors
    
    # Ground surface normal (accounts for ground slope)
    ground_normal = np.array([
        -np.sin(ground_slope_rad) * np.sin(ground_dir_rad),  # x component
        -np.sin(ground_slope_rad) * np.cos(ground_dir_rad),  # y component  
        np.cos(ground_slope_rad)                             # z component (up)
    ])
    
    # Array surface relative to ground (before considering ground slope)
    # Array tilted at array_tilt_degrees facing array_azimuth_degrees
    array_rel_normal = np.array([
        -np.sin(array_tilt_rad) * np.sin(array_azimuth_rad),
        -np.sin(array_tilt_rad) * np.cos(array_azimuth_rad),
        np.cos(array_tilt_rad)
    ])
    
    # Step 3: Combine vectors to get actual array surface normal
    # This is a simplified approach - for precise calculations, we'd need
    # full 3D rotation matrices, but this gives good approximation
    
    # Effective tilt: angle between combined normal and vertical
    # Using vector dot product: cos(angle) = v1·v2 / (|v1||v2|)
    
    # For now, use simplified geometry:
    # Effective tilt ≈ array_tilt + ground_slope_component
    effective_tilt = array_tilt_degrees + ground_slope_component
    
    # Ensure physically realistic bounds
    effective_tilt = np.clip(effective_tilt, 0, 90)
    
    # Effective azimuth (usually doesn't change much for small slopes)
    effective_azimuth = array_azimuth_degrees
    
    # Calculate correction factor
    slope_correction = effective_tilt - array_tilt_degrees
    
    return {
        'effective_tilt': effective_tilt,
        'effective_azimuth': effective_azimuth,
        'ground_slope_component': ground_slope_component,
        'slope_correction': slope_correction
    }


def analyze_array_location(terrain_file, lat, lon, array_azimuth_degrees, 
                          array_tilt_degrees):
    """
    Complete analysis of PV array geometry at a specific location.
    
    Parameters:
    -----------
    terrain_file : str
        Path to GeoJSON terrain data file
    lat, lon : float
        Array location coordinates
    array_azimuth_degrees : float
        Direction array faces (0° = North)
    array_tilt_degrees : float
        Array tilt angle relative to ground
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    # Initialize terrain analyzer
    analyzer = TerrainAnalyzer(terrain_file)
    
    # Convert coordinates
    x, y = analyzer.convert_latlon_to_utm(lat, lon)
    
    print(f"\n📍 Analyzing array location:")
    print(f"   Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"   UTM: {x:.0f}, {y:.0f}")
    print(f"   Array: {array_tilt_degrees}° tilt, {array_azimuth_degrees}° azimuth")
    
    # Get ground slope in array direction
    slope_data = analyzer.get_slope_in_direction(x, y, array_azimuth_degrees)
    
    # Calculate effective array geometry
    geometry = calculate_effective_array_geometry(
        slope_data['slope_degrees'],
        slope_data['gradient_direction'], 
        array_tilt_degrees,
        array_azimuth_degrees
    )
    
    # Combine results
    results = {
        'location': {'lat': lat, 'lon': lon, 'utm_x': x, 'utm_y': y},
        'array_config': {
            'nominal_tilt': array_tilt_degrees,
            'nominal_azimuth': array_azimuth_degrees
        },
        'ground_slope': slope_data,
        'effective_geometry': geometry,
        'correction_summary': {
            'tilt_change': geometry['slope_correction'],
            'significant_slope': abs(slope_data['slope_degrees']) > 2.0
        }
    }
    
    # Print summary
    print(f"\n📊 Analysis Results:")
    print(f"   Ground elevation: {slope_data['elevation']:.1f}m")
    print(f"   Ground slope (array direction): {slope_data['slope_degrees']:.1f}°")
    print(f"   Effective array tilt: {geometry['effective_tilt']:.1f}° (Δ{geometry['slope_correction']:+.1f}°)")
    
    if abs(geometry['slope_correction']) > 1.0:
        print(f"   ⚠️  Significant slope correction needed!")
    else:
        print(f"   ✅ Minimal slope impact")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    import os
    
    print("🔬 Ground Slope Analysis Example")
    print("=" * 50)
    
    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Current conda environment: {conda_env}")
    if conda_env != 'pv':
        print("⚠️  WARNING: Please activate the 'pv' environment first:")
        print("   conda activate pv")
        print("   python ground_slope_analysis.py")
        exit(1)
    
    # Example analysis at the Meyersdale, PA location
    terrain_file = 'data/terrain_elevation_points.geojson'
    
    # Test different array orientations
    test_orientations = [
        (180, 30),  # South-facing, 30° tilt
        (225, 30),  # Southwest-facing, 30° tilt  
        (135, 30),  # Southeast-facing, 30° tilt
    ]
    
    for azimuth, tilt in test_orientations:
        print(f"\n--- Testing {azimuth}° azimuth, {tilt}° tilt ---")
        try:
            results = analyze_array_location(
                terrain_file, LAT_CENTER, LON_CENTER, 
                azimuth, tilt
            )
        except Exception as e:
            print(f"❌ Error: {e}")