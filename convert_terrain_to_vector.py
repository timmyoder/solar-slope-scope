#!/usr/bin/env python3
"""
Convert GeoTIFF elevation data to vector format (GeoJSON) for use with geopandas
Run this script in the 'gdal' conda environment that has rasterio working

This script will:
1. Read the trimmed GeoTIFF elevation data
2. Convert raster pixels to point features with elevation values
3. Calculate slope and aspect for each point
4. Save as GeoJSON for use in the 'pv' environment with geopandas
"""

from osgeo import gdal, osr
import numpy as np
import json
import os

def convert_raster_to_vector(input_geotiff, output_geojson, sample_spacing=5):
    """
    Convert raster elevation data to vector points using GDAL
    
    Parameters:
    - input_geotiff: Path to input GeoTIFF file
    - output_geojson: Path to output GeoJSON file
    - sample_spacing: Sample every Nth pixel (5 = every 5th pixel for efficiency)
    """
    
    print(f"🔄 Converting {input_geotiff} to vector format...")
    
    # Open dataset with GDAL
    dataset = gdal.Open(input_geotiff)
    if dataset is None:
        raise ValueError(f"Could not open {input_geotiff}")
    
    # Get raster information
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    
    # Get geotransform
    geotransform = dataset.GetGeoTransform()
    
    # Get projection
    projection = dataset.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    
    # Read elevation data
    band = dataset.GetRasterBand(1)
    elevation = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    
    print(f"📊 Raster info:")
    print(f"   Size: {rows} × {cols}")
    print(f"   Projection: {srs.GetAttrValue('AUTHORITY', 1) if srs.GetAttrValue('AUTHORITY') else 'Unknown'}")
    print(f"   Elevation range: {elevation.min():.1f}m to {elevation.max():.1f}m")
    
    # Calculate pixel sizes for gradient calculation
    pixel_size_x = abs(geotransform[1])
    pixel_size_y = abs(geotransform[5])
    
    # Calculate gradients (slope components)
    dy, dx = np.gradient(elevation, pixel_size_y, pixel_size_x)
    
    # Calculate slope magnitude (in degrees)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Calculate aspect (direction of steepest slope)
    # 0° = North, 90° = East, 180° = South, 270° = West
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360
    
    # Create GeoJSON features
    features = []
    sample_count = 0
    
    # Sample points (every sample_spacing pixels for efficiency)
    for row in range(0, rows, sample_spacing):
        for col in range(0, cols, sample_spacing):
            # Skip NoData values
            elev_val = elevation[row, col]
            if np.isnan(elev_val) or (nodata_value is not None and elev_val == nodata_value):
                continue
            
            # Convert pixel coordinates to geographic coordinates
            # GDAL geotransform: [top_left_x, pixel_width, 0, top_left_y, 0, -pixel_height]
            x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
            y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
            
            # Get slope and aspect values
            slope_val = slope[row, col]
            aspect_val = aspect[row, col]
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [x, y]
                },
                "properties": {
                    "elevation": float(elev_val),
                    "slope": float(slope_val),
                    "aspect": float(aspect_val),
                    "row": int(row),
                    "col": int(col)
                }
            }
            features.append(feature)
            sample_count += 1
    
    # Get EPSG code if available
    epsg_code = srs.GetAttrValue('AUTHORITY', 1) if srs.GetAttrValue('AUTHORITY') else None
    crs_name = f"EPSG:{epsg_code}" if epsg_code else "Unknown"
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": crs_name
            }
        },
        "features": features,
        "metadata": {
            "source_file": os.path.basename(input_geotiff),
            "sample_spacing": sample_spacing,
            "pixel_size_m": [pixel_size_x, pixel_size_y],
            "elevation_range": [float(elevation.min()), float(elevation.max())],
            "total_points": sample_count
        }
    }
    
    # Save GeoJSON
    with open(output_geojson, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"   Created {sample_count:,} elevation points")
    print(f"   Saved to: {output_geojson}")
    print(f"   Sample spacing: Every {sample_spacing} pixels ({sample_spacing * pixel_size_x:.1f}m)")
    
    # Clean up
    dataset = None

def main():
    # File paths
    input_file = "data/USGS_1M_17_x66y441_PA_WesternPA_2019_D20_trimmed_500m.tif"
    output_file = "data/terrain_elevation_points.geojson"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Available files in data/:")
        if os.path.exists("data"):
            for f in os.listdir("data"):
                if f.endswith(('.tif', '.tiff')):
                    print(f"   {f}")
        return
    
    # Convert with different sample spacings based on file size
    try:
        # First, check the raster size to determine appropriate sampling
        dataset = gdal.Open(input_file)
        if dataset is None:
            print(f"❌ Could not open {input_file}")
            return
            
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        total_pixels = rows * cols
        dataset = None  # Close dataset
        
        print(f"📏 Raster size: {rows} × {cols} = {total_pixels:,} pixels")
        
        # Determine sample spacing to keep output manageable
        if total_pixels < 10000:  # Small raster
            sample_spacing = 1  # Every pixel
        elif total_pixels < 100000:  # Medium raster
            sample_spacing = 3  # Every 3rd pixel
        else:  # Large raster
            sample_spacing = 5  # Every 5th pixel
            
        print(f"🎯 Using sample spacing: {sample_spacing} (for manageable output size)")
        
        # Convert
        convert_raster_to_vector(input_file, output_file, sample_spacing)
        
        print(f"\n🎉 Success! You can now use '{output_file}' in your 'pv' environment with:")
        print(f"   import geopandas as gpd")
        print(f"   terrain_gdf = gpd.read_file('{output_file}')")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()