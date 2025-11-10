#!/usr/bin/env python3
"""
Simple GeoTIFF Trimmer (Alternative approach without rasterio)
Uses GDAL command line tools to trim elevation data
"""

import os
import subprocess
import sys

def check_gdal_installed():
    """Check if GDAL tools are available"""
    try:
        result = subprocess.run(['gdalinfo', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GDAL found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("GDAL not found. Please install with:")
    print("  conda install -c conda-forge gdal")
    return False

def trim_with_gdal(input_file: str, 
                  center_lat: float, 
                  center_lon: float,
                  radius_meters: float = 1000,
                  output_file: str = None) -> str:
    """
    Trim GeoTIFF using GDAL command line tools
    """
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_trimmed_{radius_meters}m.tif"
    
    # Convert radius to degrees (approximate)
    radius_deg = radius_meters / 111000.0
    
    # Calculate bounding box
    min_lon = center_lon - radius_deg
    max_lon = center_lon + radius_deg
    min_lat = center_lat - radius_deg  
    max_lat = center_lat + radius_deg
    
    print(f"Trimming {input_file}")
    print(f"Center: {center_lat:.6f}, {center_lon:.6f}")
    print(f"Radius: {radius_meters}m")
    print(f"Bounding box: {min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
    
    # Use gdal_translate to extract the subset
    cmd = [
        'gdal_translate',
        '-projwin', str(min_lon), str(max_lat), str(max_lon), str(min_lat),
        '-projwin_srs', 'EPSG:4326',  # WGS84
        input_file,
        output_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success! Trimmed file saved as: {output_file}")
            
            # Get info about the output file
            info_cmd = ['gdalinfo', output_file]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            
            if info_result.returncode == 0:
                lines = info_result.stdout.split('\n')
                for line in lines:
                    if 'Size is' in line:
                        print(f"Output size: {line.strip()}")
                    elif 'Upper Left' in line or 'Lower Right' in line:
                        print(f"Bounds: {line.strip()}")
            
            return output_file
        else:
            print(f"❌ Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error running gdal_translate: {e}")
        return None

def list_elevation_files():
    """List elevation files in data directory"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found")
        return []
    
    tif_files = [f for f in os.listdir(data_dir) 
                 if f.lower().endswith(('.tif', '.tiff'))]
    
    print(f"Elevation files in {data_dir}:")
    for i, file in enumerate(tif_files):
        filepath = os.path.join(data_dir, file)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {i+1}. {file} ({file_size_mb:.1f} MB)")
    
    return tif_files

def main():
    """Interactive trimming using GDAL"""
    
    print("Simple GeoTIFF Elevation Data Trimmer (GDAL version)")
    print("=" * 60)
    
    # Check if GDAL is available
    if not check_gdal_installed():
        return
    
    # List available files
    tif_files = list_elevation_files()
    if not tif_files:
        print("No .tif files found in data directory!")
        return
    
    # Your coordinates (Meyersdale, PA)
    center_lat = 39.796679
    center_lon = -79.092431
    
    print(f"\nTarget location: {center_lat}, {center_lon} (Meyersdale, PA)")
    
    # Select file
    if len(tif_files) == 1:
        input_file = os.path.join("data", tif_files[0])
        print(f"Using file: {tif_files[0]}")
    else:
        while True:
            try:
                choice = int(input("Select file number: ")) - 1
                if 0 <= choice < len(tif_files):
                    input_file = os.path.join("data", tif_files[choice])
                    break
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a number!")
    
    # Select radius
    radius_options = [
        (500, "Small area - 500m radius (1km diameter) - for detailed constraint analysis"),
        (1000, "Medium area - 1km radius (2km diameter) - good for general site analysis"),
        (2000, "Large area - 2km radius (4km diameter) - for regional context")
    ]
    
    print("\nChoose extraction radius:")
    for i, (radius, description) in enumerate(radius_options):
        print(f"  {i+1}. {description}")
    
    while True:
        try:
            choice = int(input("Select radius option: ")) - 1
            if 0 <= choice < len(radius_options):
                radius_meters = radius_options[choice][0]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    # Trim the file
    output_file = trim_with_gdal(
        input_file, 
        center_lat, 
        center_lon,
        radius_meters
    )
    
    if output_file:
        print(f"\n✅ Success! Trimmed file ready: {output_file}")
        print(f"\nNext steps:")
        print(f"1. Use this trimmed file with constrained_area_analysis.py")
        print(f"2. Command: python constrained_area_analysis.py {output_file} {center_lat} {center_lon} {radius_meters//2}")
        print(f"3. Or use with local_topography_analysis.py for detailed analysis")

if __name__ == "__main__":
    main()