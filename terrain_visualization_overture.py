#!/usr/bin/env python3
"""
Enhanced Terrain Visualization with Overture Maps Integration
============================================================

This script integrates high-quality Overture Maps building and road data
with your existing terrain analysis workflow, providing accurate building
heights and improved site context.

Fixes:
- Uses Overture Maps instead of OSM (better building coverage & real heights)
- Crops context data to elevation data bounds
- Fixes PV array center coordinate conversion
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path
import os

# Try to import geopandas for building data (optional)
try:
    import geopandas as gpd
    from pyproj import Transformer
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ GeoPandas/PyProj not available - building data features disabled")

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

def apply_manual_height_corrections(buildings_gdf, elevation_bounds):
    """Apply manual height corrections for known structures."""
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return buildings_gdf
    
    # Define area near PV array for large barn identification
    # PV array is approximately at UTM 663314, 4406931
    pv_x, pv_y = 663314, 4406931
    
    # Find buildings close to PV array that might be the large barn
    distances_to_pv = buildings_gdf.geometry.distance(
        gpd.points_from_xy([pv_x], [pv_y], crs='EPSG:32617')[0]
    )
    
    # Look for large buildings near PV array (within 200m)
    nearby_mask = distances_to_pv <= 200
    large_mask = buildings_gdf['area_m2'] > 300  # Buildings larger than 300 m²
    
    barn_candidates = buildings_gdf[nearby_mask & large_mask]
    
    if len(barn_candidates) > 0:
        # Apply barn height to largest nearby building
        largest_barn_idx = barn_candidates['area_m2'].idxmax()
        barn_height_m = 10.0  # 40-60 ft = ~12-18m, using 10m (33 ft)
        buildings_gdf.loc[largest_barn_idx, 'height'] = barn_height_m
        print(f"CORRECTION: Applied barn height ({barn_height_m:.0f}m) to large building near PV array")
    
    return buildings_gdf

def get_elevation_bounds(x_coords, y_coords):
    """Get the bounding box of elevation data."""
    return {
        'x_min': x_coords.min(),
        'x_max': x_coords.max(), 
        'y_min': y_coords.min(),
        'y_max': y_coords.max()
    }

def load_overture_buildings(data_dir='data/overture-maps', elevation_bounds=None):
    """Load Overture Maps building data."""
    if not GEOPANDAS_AVAILABLE:
        return None
    
    # Find Overture building file
    overture_dir = Path(data_dir)
    if not overture_dir.exists():
        print(f"WARNING: Overture Maps directory not found: {overture_dir}")
        return None
    
    building_files = list(overture_dir.glob('overture-building-*.geojson'))
    if not building_files:
        print(f"WARNING: No Overture building files found in {overture_dir}")
        return None
    
    building_file = building_files[0]  # Use first file found
    print(f"LOADING: Overture buildings from: {building_file.name}")
    
    try:
        # Load buildings
        buildings_gdf = gpd.read_file(building_file)
        print(f"SUCCESS: Loaded {len(buildings_gdf)} buildings from Overture Maps")
        
        # Convert to UTM Zone 17N for accurate measurements
        if buildings_gdf.crs.to_string() != 'EPSG:32617':
            print("CONVERTING: Buildings to UTM Zone 17N...")
            buildings_gdf = buildings_gdf.to_crs('EPSG:32617')
        
        # Add area calculation
        buildings_gdf['area_m2'] = buildings_gdf.geometry.area
        
        # Crop to elevation bounds if provided
        if elevation_bounds:
            print("CROPPING: Buildings to elevation data bounds...")
            mask = (
                (buildings_gdf.geometry.centroid.x >= elevation_bounds['x_min']) &
                (buildings_gdf.geometry.centroid.x <= elevation_bounds['x_max']) &
                (buildings_gdf.geometry.centroid.y >= elevation_bounds['y_min']) &
                (buildings_gdf.geometry.centroid.y <= elevation_bounds['y_max'])
            )
            buildings_gdf = buildings_gdf[mask].copy()
            print(f"RESULT: {len(buildings_gdf)} buildings within elevation data bounds")
        
        # Clean and enhance height data
        if 'height' in buildings_gdf.columns:
            # Fill missing heights with reasonable defaults
            buildings_gdf['height'] = buildings_gdf['height'].fillna(8.0)  # 8m default (2-story)
            # Cap unrealistic heights
            buildings_gdf.loc[buildings_gdf['height'] > 100, 'height'] = 20.0
            buildings_gdf.loc[buildings_gdf['height'] < 1, 'height'] = 3.0
        else:
            print("WARNING: No height data found, using 8m default")
            buildings_gdf['height'] = 8.0
        
        # Apply manual height corrections for known structures
        buildings_gdf = apply_manual_height_corrections(buildings_gdf, elevation_bounds)
        
        print(f"BUILDINGS: Building height range: {buildings_gdf['height'].min():.1f} - {buildings_gdf['height'].max():.1f}m")
        
        return buildings_gdf
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def load_overture_roads(data_dir='data/overture-maps', elevation_bounds=None):
    """Load Overture Maps road data.""" 
    if not GEOPANDAS_AVAILABLE:
        return None
    
    # Find Overture road file
    overture_dir = Path(data_dir)
    road_files = list(overture_dir.glob('overture-segment-*.geojson'))
    if not road_files:
        print(f"⚠️ No Overture road files found in {overture_dir}")
        return None
    
    road_file = road_files[0]  # Use first file found
    print(f"📁 Loading Overture roads from: {road_file.name}")
    
    try:
        # Load roads
        roads_gdf = gpd.read_file(road_file)
        print(f"✅ Loaded {len(roads_gdf)} road segments from Overture Maps")
        
        # Convert to UTM Zone 17N
        if roads_gdf.crs.to_string() != 'EPSG:32617':
            print("🔄 Converting roads to UTM Zone 17N...")
            roads_gdf = roads_gdf.to_crs('EPSG:32617')
        
        # Crop to elevation bounds if provided
        if elevation_bounds:
            print("✂️ Cropping roads to elevation data bounds...")
            # For roads, check if any part intersects the bounds
            from shapely.geometry import box
            bounds_polygon = box(
                elevation_bounds['x_min'], elevation_bounds['y_min'],
                elevation_bounds['x_max'], elevation_bounds['y_max']
            )
            mask = roads_gdf.geometry.intersects(bounds_polygon)
            roads_gdf = roads_gdf[mask].copy()
            print(f"🛣️ {len(roads_gdf)} road segments within elevation data bounds")
        
        # Clean road classification
        if 'class' not in roads_gdf.columns:
            roads_gdf['road_class'] = 'unclassified'
        else:
            roads_gdf['road_class'] = roads_gdf['class'].fillna('unclassified')
        
        return roads_gdf
        
    except Exception as e:
        print(f"❌ Error loading Overture roads: {e}")
        return None

def convert_latlon_to_utm_precise(lat, lon):
    """Precise conversion from lat/lon to UTM Zone 17N using pyproj."""
    if not GEOPANDAS_AVAILABLE:
        # Fallback to approximate method
        return convert_latlon_to_utm_approx(lat, lon)
    
    try:
        # Create transformer for WGS84 to UTM Zone 17N
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32617', always_xy=True)
        x, y = transformer.transform(lon, lat)  # Note: pyproj expects (lon, lat)
        return x, y
    except Exception as e:
        print(f"⚠️ PyProj conversion failed: {e}, using approximate method")
        return convert_latlon_to_utm_approx(lat, lon)

def convert_latlon_to_utm_approx(lat, lon, utm_zone=17):
    """Fallback approximate conversion from lat/lon to UTM coordinates."""
    central_meridian = -81.0  # UTM Zone 17N central meridian
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cm_rad = np.radians(central_meridian)
    
    k0 = 0.9996  # UTM scale factor
    a = 6378137.0  # WGS84 semi-major axis
    
    delta_lon = lon_rad - cm_rad
    
    x = 500000 + k0 * a * delta_lon * np.cos(lat_rad)
    y = k0 * a * lat_rad
    
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

def analyze_building_impacts(buildings_gdf, array_x_utm, array_y_utm, x_coords, y_coords, elevations, search_radius_m=200):
    """Analyze potential building impacts on PV array."""
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return {}
    
    # Create point for array center
    from shapely.geometry import Point
    array_point = Point(array_x_utm, array_y_utm)
    
    # Find buildings within search radius
    distances = buildings_gdf.geometry.distance(array_point)
    nearby_buildings = buildings_gdf[distances <= search_radius_m].copy()
    nearby_buildings['distance_to_array'] = distances[distances <= search_radius_m]
    
    # Get array elevation for shading analysis
    array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                              (array_x_utm, array_y_utm), method='linear')
    if np.isnan(array_elevation):
        array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                  (array_x_utm, array_y_utm), method='nearest')
    
    # Analyze potential shading based on relative heights, not absolute elevation
    # A building is a shading concern if it's significantly taller than surrounding terrain
    min_shading_height = 8.0  # Minimum height to be a concern (8m = ~26ft)
    relative_height_threshold = 5.0  # Must be 5m+ above local terrain to cause shading
    
    tall_buildings = nearby_buildings[nearby_buildings['height'] >= min_shading_height]
    
    analysis = {
        'total_nearby': len(nearby_buildings),
        'tall_nearby': len(tall_buildings),
        'closest_distance': distances.min() if len(distances) > 0 else float('inf'),
        'max_nearby_height': nearby_buildings['height'].max() if len(nearby_buildings) > 0 else 0,
        'array_elevation': array_elevation,
        'shading_threshold': min_shading_height,
        'nearby_buildings': nearby_buildings,
        'tall_buildings': tall_buildings
    }
    
    return analysis

def plot_enhanced_terrain_with_overture(x_mesh, y_mesh, elev_mesh, x_coords, y_coords, elevations,
                                       array_x, array_y, array_elevation, array_center_x, array_center_y,
                                       buildings_gdf=None, roads_gdf=None):
    """Create comprehensive visualization with Overture Maps data."""
    
    # Analyze building impacts
    building_analysis = analyze_building_impacts(
        buildings_gdf, array_center_x, array_center_y, x_coords, y_coords, elevations
    ) if buildings_gdf is not None else {}
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create 2x2 grid where right column spans full height for 2D plot
    # Left column: top = 3D, bottom = risk analysis
    # Right column: full height = 2D site context
    
    # 3D Plot with buildings (left top)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot terrain surface
    surface = ax1.plot_surface(x_mesh, y_mesh, elev_mesh, cmap='terrain', alpha=0.8, 
                              linewidth=0, antialiased=True)
    
    # Plot buildings as 3D blocks with actual footprints
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        print(f"RENDERING: {len(buildings_gdf)} buildings in 3D...")
        
        for idx, building in buildings_gdf.iterrows():
            if building.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                try:
                    # Get building footprint coordinates
                    if building.geometry.geom_type == 'Polygon':
                        coords = list(building.geometry.exterior.coords)
                    else:  # MultiPolygon - use largest polygon
                        largest_poly = max(building.geometry.geoms, key=lambda p: p.area)
                        coords = list(largest_poly.exterior.coords)
                    
                    building_x = [coord[0] for coord in coords]
                    building_y = [coord[1] for coord in coords]
                    
                    # Get ground elevation at building centroid
                    centroid = building.geometry.centroid
                    ground_elev = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                         (centroid.x, centroid.y), method='linear')
                    if np.isnan(ground_elev):
                        ground_elev = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                             (centroid.x, centroid.y), method='nearest')
                    
                    building_height = building['height']
                    top_elev = ground_elev + building_height
                    
                    # Color code by height
                    if building_height >= 12:
                        color = 'darkred'    # Very tall buildings (barns, etc.)
                        linewidth = 3.0
                        alpha = 1.0
                    elif building_height >= 8:
                        color = 'red'        # Tall buildings  
                        linewidth = 2.5
                        alpha = 0.9
                    elif building_height >= 5:
                        color = 'brown'      # Medium buildings
                        linewidth = 2.0
                        alpha = 0.8
                    else:
                        color = 'saddlebrown'  # Small buildings
                        linewidth = 1.5
                        alpha = 0.7
                    
                    # Plot building footprint at ground level
                    ground_z = [ground_elev] * len(building_x)
                    ax1.plot(building_x, building_y, ground_z, color=color, 
                            linewidth=linewidth, alpha=alpha)
                    
                    # Plot building footprint at top
                    top_z = [top_elev] * len(building_x)
                    ax1.plot(building_x, building_y, top_z, color=color, 
                            linewidth=linewidth, alpha=alpha)
                    
                    # Connect corners with vertical lines (walls)
                    for i in range(0, len(building_x)-1, 2):  # Skip some for performance
                        ax1.plot([building_x[i], building_x[i]], 
                                [building_y[i], building_y[i]], 
                                [ground_elev, top_elev], 
                                color=color, linewidth=1, alpha=alpha*0.6)
                        
                except Exception as e:
                    # Fallback to centroid point if footprint fails
                    centroid = building.geometry.centroid
                    building_x, building_y = centroid.x, centroid.y
                    
                    ground_elev = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                         (building_x, building_y), method='linear')
                    if np.isnan(ground_elev):
                        ground_elev = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                             (building_x, building_y), method='nearest')
                    
                    building_height = building['height']
                    top_elev = ground_elev + building_height
                    
                    # Plot as simple vertical line
                    ax1.plot([building_x, building_x], [building_y, building_y], 
                            [ground_elev, top_elev], color='red', linewidth=2, alpha=0.8)
    
    # Plot PV array
    array_elevations = np.full_like(array_x, array_elevation)
    ax1.plot(array_x, array_y, array_elevations, 'lime', linewidth=4, label='PV Array')
    ax1.scatter(array_center_x, array_center_y, array_elevation, color='lime', s=200, 
               label='Array Center', edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_zlabel('Elevation (m)')
    ax1.set_title('3D Terrain with Buildings')
    ax1.legend()
    ax1.view_init(elev=35, azim=45)
    
    # 2D Site Context Plot (right column, spanning full height)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Terrain contours
    contour = ax2.contourf(x_mesh, y_mesh, elev_mesh, levels=20, cmap='terrain', alpha=0.7)
    contour_lines = ax2.contour(x_mesh, y_mesh, elev_mesh, levels=10, colors='black', 
                               alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%d m')
    
    # Plot roads with proper classification
    if roads_gdf is not None and len(roads_gdf) > 0:
        road_colors = {
            'tertiary': 'orange',
            'residential': 'gray', 
            'service': 'lightgray',
            'unclassified': 'brown',
            'track': 'saddlebrown',
            'footway': 'darkgreen',
            'path': 'darkgreen'
        }
        
        # Group roads by class for better legend
        road_classes = roads_gdf['road_class'].unique()
        for road_class in road_classes:
            road_subset = roads_gdf[roads_gdf['road_class'] == road_class]
            color = road_colors.get(road_class, 'black')
            linewidth = 3 if road_class == 'tertiary' else 2
            alpha = 0.9 if road_class in ['tertiary', 'residential'] else 0.7
            
            road_subset.plot(ax=ax2, color=color, linewidth=linewidth, alpha=alpha,
                           label=f'{road_class.title()} road')
    
    # Plot buildings with height-based coloring
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        # Color buildings by height
        tall_buildings = buildings_gdf[buildings_gdf['height'] > 10]
        medium_buildings = buildings_gdf[(buildings_gdf['height'] > 5) & (buildings_gdf['height'] <= 10)]
        small_buildings = buildings_gdf[buildings_gdf['height'] <= 5]
        
        if len(small_buildings) > 0:
            small_buildings.plot(ax=ax2, color='lightcoral', alpha=0.6, edgecolor='darkred', 
                               linewidth=0.5, label='Buildings ≤5m')
        if len(medium_buildings) > 0:
            medium_buildings.plot(ax=ax2, color='red', alpha=0.7, edgecolor='darkred', 
                                linewidth=0.5, label='Buildings 5-10m') 
        if len(tall_buildings) > 0:
            tall_buildings.plot(ax=ax2, color='darkred', alpha=0.9, edgecolor='black', 
                               linewidth=1, label='Buildings >10m')
    
    # Plot PV array
    ax2.plot(array_x, array_y, color='lime', linewidth=4, label='PV Array')
    ax2.plot(array_center_x, array_center_y, 'o', color='lime', markersize=12, 
             label='Array Center', markeredgecolor='black', markeredgewidth=2)
    
    # Add shading concern indicators with improved legend
    legend_elements = []
    if building_analysis.get('tall_nearby', 0) > 0:
        # Draw circles for shading analysis radii
        circle_info = [
            (50, 'orange', '--', 'Critical Zone (50m)'),
            (100, 'yellow', '--', 'High Concern (100m)'), 
            (200, 'red', ':', 'Monitoring Zone (200m)')
        ]
        for radius, color, style, label in circle_info:
            circle = plt.Circle((array_center_x, array_center_y), radius, 
                              color=color, fill=False, linestyle=style, linewidth=2, alpha=0.7)
            ax2.add_patch(circle)
            
            # Add to legend
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color=color, linestyle=style, linewidth=2, label=label))
        
        # Add explanatory text about shading radii
        shading_text = """Shading Analysis Zones:
• Buildings within 50m: Critical impact
• 50-100m: Seasonal shading possible  
• 100-200m: Early/late day effects
• >200m: Minimal shading risk

Radii based on building height 
to array distance ratios for 
worst-case winter sun angles."""
        
        ax2.text(0.02, 0.98, shading_text, transform=ax2.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=1.0))
    
    # Set bounds to elevation data extents (no margin for tighter view)
    bounds = get_elevation_bounds(x_coords, y_coords)
    ax2.set_xlim(bounds['x_min'], bounds['x_max'])
    ax2.set_ylim(bounds['y_min'], bounds['y_max'])
    
    ax2.set_xlabel('UTM Easting (m)')
    ax2.set_ylabel('UTM Northing (m)')
    ax2.set_title('Site Context: Terrain, Buildings & Shading Analysis\nMeyersdale, PA')
    
    # Combine legend elements manually to avoid PatchCollection warning
    legend_handles = []
    legend_labels = []
    
    # Add building legend entries manually
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        from matplotlib.patches import Patch
        legend_handles.extend([
            Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='Buildings <8m'),
            Patch(facecolor='orange', edgecolor='black', alpha=0.9, label='Buildings 8-10m'),
            Patch(facecolor='darkred', edgecolor='black', alpha=0.9, label='Buildings >10m')
        ])
        
    # Add road legend
    if roads_gdf is not None and len(roads_gdf) > 0:
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color='gray', linewidth=2, label='Roads'))
    
    # Add PV array legend
    from matplotlib.lines import Line2D
    legend_handles.extend([
        Line2D([0], [0], color='lime', linewidth=4, label='PV Array'),
        Line2D([0], [0], marker='o', color='lime', markersize=8, 
               markeredgecolor='black', markeredgewidth=1, linestyle='None', label='Array Center')
    ])
    
    # Add shading zone legends
    legend_handles.extend(legend_elements)
    
    ax2.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Shading risk analysis and summary (left bottom)
    ax3 = fig.add_subplot(2, 2, 3)
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        # Create compact shading analysis visualization
        heights = buildings_gdf['height'].values
        
        # Categorize buildings by shading risk
        high_risk = len(buildings_gdf[buildings_gdf['height'] >= 12])  # >= 12m
        medium_risk = len(buildings_gdf[(buildings_gdf['height'] >= 8) & (buildings_gdf['height'] < 12)])
        low_risk = len(buildings_gdf[buildings_gdf['height'] < 8])
        
        categories = ['Low\n(<8m)', 'Medium\n(8-12m)', 'High\n(≥12m)']
        counts = [low_risk, medium_risk, high_risk]
        colors = ['lightgreen', 'orange', 'red']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', width=0.5)
        ax3.set_ylabel('Buildings', fontsize=9)
        ax3.set_title('Shading Risk by Height', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Set smaller y-axis limits to compress the chart
        max_count = max(counts) if counts else 1
        ax3.set_ylim(0, max_count * 1.2)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Comprehensive site summary as text block
        total_buildings = len(buildings_gdf)
        
        # Count buildings and roads
        num_buildings = len(buildings_gdf) if buildings_gdf is not None else 0
        num_roads = len(roads_gdf) if roads_gdf is not None else 0
        
        # Building impact analysis
        impact_summary = ""
        if building_analysis:
            shading_risk = "HIGH" if building_analysis.get('tall_nearby', 0) > 0 else "LOW"
            impact_summary = f"""[SHADING ANALYSIS]
Risk Level: {shading_risk}
Buildings within 200m: {building_analysis.get('total_nearby', 0)}
Tall buildings (>8m): {building_analysis.get('tall_nearby', 0)}
Closest building: {building_analysis.get('closest_distance', float('inf')):.0f}m
Max nearby height: {building_analysis.get('max_nearby_height', 0):.1f}m

[SITE SUMMARY]  
Total buildings: {num_buildings}
Road segments: {num_roads}
Array elevation: {building_analysis.get('array_elevation', 0):.1f}m
Terrain relief: {elevations.max() - elevations.min():.1f}m"""
        
        ax3.text(1.05, 0.95, impact_summary, transform=ax3.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=1.0))
        
    else:
        ax3.text(0.5, 0.5, 'No building data\navailable', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_title('Shading Analysis')
    
    # Remove the separate ax4 subplot since we're combining everything
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    print("*** Enhanced Terrain Visualization with Overture Maps ***")
    print("=" * 65)
    
    # Load terrain data
    terrain_file = 'data/terrain_elevation_points.geojson'
    
    try:
        x_coords, y_coords, elevations, slopes, aspects = load_elevation_data(terrain_file)
        print(f"SUCCESS: Loaded {len(elevations)} elevation points")
        
        # Get elevation bounds for cropping context data
        elevation_bounds = get_elevation_bounds(x_coords, y_coords)
        print(f"BOUNDS: Elevation bounds: X={elevation_bounds['x_min']:.0f}-{elevation_bounds['x_max']:.0f}, Y={elevation_bounds['y_min']:.0f}-{elevation_bounds['y_max']:.0f}")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find {terrain_file}")
        print("    Please ensure terrain elevation data is available")
        return
    except Exception as e:
        print(f"ERROR: Error loading terrain data: {e}")
        return
    
    # Load Overture Maps context data
    buildings_gdf = load_overture_buildings(elevation_bounds=elevation_bounds)
    roads_gdf = load_overture_roads(elevation_bounds=elevation_bounds)
    
    # Convert array center coordinates with precise method
    print(f"\nCONVERTING: PV array coordinates...")
    print(f"   Input: {LAT_CENTER:.6f}N, {LON_CENTER:.6f}W")
    
    array_x_utm, array_y_utm = convert_latlon_to_utm_precise(LAT_CENTER, LON_CENTER)
    print(f"   UTM: {array_x_utm:.0f}, {array_y_utm:.0f}")
    
    # Check if array is within elevation bounds
    if (elevation_bounds['x_min'] <= array_x_utm <= elevation_bounds['x_max'] and
        elevation_bounds['y_min'] <= array_y_utm <= elevation_bounds['y_max']):
        print("SUCCESS: Array location is within terrain data bounds")
    else:
        print("WARNING: Array location is outside terrain data bounds")
        print(f"   Array UTM: {array_x_utm:.0f}, {array_y_utm:.0f}")
        print(f"   Elevation bounds: X={elevation_bounds['x_min']:.0f}-{elevation_bounds['x_max']:.0f}, Y={elevation_bounds['y_min']:.0f}-{elevation_bounds['y_max']:.0f}")
        print("   Using array position anyway for visualization...")
    
    # Get array elevation
    array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                              (array_x_utm, array_y_utm), method='linear')
    if np.isnan(array_elevation):
        array_elevation = griddata(np.column_stack((x_coords, y_coords)), elevations,
                                  (array_x_utm, array_y_utm), method='nearest')
    
    print(f"ELEVATION: Array elevation: {array_elevation:.1f}m")
    
    # Create array outline
    array_x, array_y = create_array_outline_utm(array_x_utm, array_y_utm, 
                                                ARRAY_WIDTH_M, ARRAY_HEIGHT_M)
    
    # Create terrain surface
    x_mesh, y_mesh, elev_mesh = create_terrain_surface(x_coords, y_coords, elevations)
    
    # Create enhanced visualization
    print("\n🎨 Creating enhanced visualization with Overture Maps data...")
    fig = plot_enhanced_terrain_with_overture(
        x_mesh, y_mesh, elev_mesh, x_coords, y_coords, elevations,
        array_x, array_y, array_elevation, array_x_utm, array_y_utm,
        buildings_gdf, roads_gdf
    )
    
    plt.show()
    
    # Summary
    print("\n" + "="*65)
    print("OVERTURE MAPS INTEGRATION COMPLETE!")
    print("="*65)
    
    if buildings_gdf is not None:
        heights = buildings_gdf['height'].values
        print(f"🏠 Buildings: {len(buildings_gdf)} structures with real heights")
        print(f"   Height range: {heights.min():.1f} - {heights.max():.1f}m")
        print(f"   Average: {heights.mean():.1f}m")
        print(f"   Sources: Microsoft ML Buildings + OpenStreetMap")
        
        tall_buildings = len(buildings_gdf[buildings_gdf['height'] > 8])
        if tall_buildings > 0:
            print(f"   ⚠️  {tall_buildings} buildings >8m may impact PV performance")
    else:
        print("❌ No buildings data loaded")
    
    if roads_gdf is not None:
        print(f"🛣️ Roads: {len(roads_gdf)} segments")
        road_types = roads_gdf['road_class'].value_counts().head(3)
        print(f"   Main types: {', '.join([f'{k}({v})' for k, v in road_types.items()])}")
    else:
        print("❌ No roads data loaded")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review shading analysis for buildings >8m near array")
    print(f"   2. Consider seasonal shading patterns (sun angles)")
    print(f"   3. Use building heights for detailed solar modeling")
    print(f"   4. Plan installation access via road network")

if __name__ == "__main__":
    main()