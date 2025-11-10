#!/usr/bin/env python3
"""
Quick data inspection script to understand the GeoJSON structure
"""

import json

def inspect_geojson(file_path):
    """Inspect the structure of the GeoJSON file"""
    print("Inspecting GeoJSON file structure...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Type: {data.get('type', 'Unknown')}")
    print(f"Number of features: {len(data.get('features', []))}")
    
    # Look at first feature
    if 'features' in data and len(data['features']) > 0:
        first_feature = data['features'][0]
        print(f"\nFirst feature structure:")
        print(f"  Geometry type: {first_feature.get('geometry', {}).get('type', 'Unknown')}")
        print(f"  Properties keys: {list(first_feature.get('properties', {}).keys())}")
        
        # Check coordinates
        coords = first_feature.get('geometry', {}).get('coordinates', [])
        print(f"  Coordinates structure: {len(coords)} dimensions")
        if coords:
            print(f"  Example coordinates: {coords}")
        
        # Check properties
        props = first_feature.get('properties', {})
        print(f"  Example properties: {props}")

if __name__ == "__main__":
    try:
        inspect_geojson('data/terrain_elevation_points.geojson')
    except FileNotFoundError:
        print("Error: Could not find data/terrain_elevation_points.geojson")
    except Exception as e:
        print(f"Error: {e}")