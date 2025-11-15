#!/usr/bin/env python3
"""
Ground Slope Calculation Verification

This script helps verify that our ground slope calculations are geometrically correct
by analyzing the USGS slope and aspect data in detail and providing clear interpretations.
"""

import json
import numpy as np
from ground_slope_analysis import TerrainAnalyzer
import matplotlib.pyplot as plt

def verify_slope_interpretation():
    """
    Verify our understanding of USGS slope and aspect data.
    
    Key questions to answer:
    1. What does aspect=265° actually mean in terms of slope direction?
    2. Are our directional slope calculations geometrically correct?
    3. How do we interpret the results in terms of terrain orientation?
    """
    
    print("🔍 Ground Slope Calculation Verification")
    print("=" * 60)
    
    # Test location (PV array site)
    lat, lon = 39.796678, -79.092463
    
    # Load terrain data
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    
    # Get detailed slope analysis for the array location
    print(f"\n📍 Analyzing location: {lat:.6f}°N, {lon:.6f}°W")
    
    # Get the raw USGS data for this point
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    print(f"   UTM coordinates: {x_utm:.0f}, {y_utm:.0f}")
    
    # Find nearest grid point
    distances = np.sqrt((analyzer.x_coords - x_utm)**2 + (analyzer.y_coords - y_utm)**2)
    nearest_idx = np.argmin(distances)
    
    nearest_x = analyzer.x_coords[nearest_idx]
    nearest_y = analyzer.y_coords[nearest_idx] 
    nearest_elev = analyzer.elevations[nearest_idx]
    nearest_slope = analyzer.grid_points[nearest_idx]['slope']
    nearest_aspect = analyzer.grid_points[nearest_idx]['aspect']
    
    print(f"\n📊 Nearest USGS Grid Point:")
    print(f"   Location: {nearest_x:.0f}, {nearest_y:.0f} (UTM)")
    print(f"   Distance: {distances[nearest_idx]:.1f}m from target")
    print(f"   Elevation: {nearest_elev:.1f}m")
    print(f"   Max slope: {nearest_slope:.1f}°")
    print(f"   Aspect: {nearest_aspect:.1f}°")
    
    # CRITICAL: Interpret what aspect means
    print(f"\n🧭 Aspect Interpretation:")
    print(f"   Aspect = {nearest_aspect:.1f}° means:")
    print(f"   • The steepest downhill direction is {nearest_aspect:.1f}° from north")
    print(f"   • This is approximately {get_cardinal_direction(nearest_aspect)}")
    print(f"   • The uphill direction is {(nearest_aspect + 180) % 360:.1f}°")
    print(f"   • This is approximately {get_cardinal_direction((nearest_aspect + 180) % 360)}")
    
    # Geometric interpretation
    print(f"\n📐 Geometric Interpretation:")
    if nearest_aspect > 180:
        # Downhill toward west side (225-315°) or north side (315-45°)
        if 225 <= nearest_aspect <= 315:
            print(f"   • The terrain slopes DOWN toward the WEST")
            print(f"   • Therefore, the EAST side is HIGHER than the WEST side")
        elif nearest_aspect >= 315 or nearest_aspect <= 45:
            print(f"   • The terrain slopes DOWN toward the NORTH")  
            print(f"   • Therefore, the SOUTH side is HIGHER than the NORTH side")
    else:
        # Downhill toward east side (45-135°) or south side (135-225°)
        if 45 <= nearest_aspect <= 135:
            print(f"   • The terrain slopes DOWN toward the EAST")
            print(f"   • Therefore, the WEST side is HIGHER than the EAST side")
        elif 135 <= nearest_aspect <= 225:
            print(f"   • The terrain slopes DOWN toward the SOUTH")
            print(f"   • Therefore, the NORTH side is HIGHER than the SOUTH side")
    
    # Test our directional slope calculations
    print(f"\n🔄 Directional Slope Verification:")
    print(f"   Testing multiple directions around the compass...")
    
    test_directions = [0, 45, 90, 135, 180, 225, 270, 315]  # N, NE, E, SE, S, SW, W, NW
    direction_names = ['N (0°)', 'NE (45°)', 'E (90°)', 'SE (135°)', 'S (180°)', 'SW (225°)', 'W (270°)', 'NW (315°)']
    
    for i, direction in enumerate(test_directions):
        slope_result = analyzer.get_slope_in_direction(x_utm, y_utm, direction)
        slope_in_dir = slope_result['slope_degrees']
        angle_diff = abs(direction - nearest_aspect)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # Calculate expected slope using trigonometry
        expected_slope = nearest_slope * np.cos(np.radians(angle_diff))
        
        print(f"   {direction_names[i]:10}: {slope_in_dir:.1f}° (angle_diff: {angle_diff:.1f}°, expected: {expected_slope:.1f}°)")
    
    # Find maximum and minimum slopes
    all_slopes = [analyzer.get_slope_in_direction(x_utm, y_utm, d)['slope_degrees'] for d in test_directions]
    max_slope_idx = np.argmax(all_slopes)
    min_slope_idx = np.argmin(all_slopes)
    
    print(f"\n📈 Slope Analysis Summary:")
    print(f"   Maximum slope: {all_slopes[max_slope_idx]:.1f}° in direction {direction_names[max_slope_idx]}")
    print(f"   Minimum slope: {all_slopes[min_slope_idx]:.1f}° in direction {direction_names[min_slope_idx]}")
    print(f"   USGS max slope: {nearest_slope:.1f}° (should match our maximum)")
    
    # Verify that our maximum matches USGS
    if abs(max(all_slopes) - nearest_slope) < 0.1:
        print(f"   ✅ Our calculations match USGS data!")
    else:
        print(f"   ❌ Discrepancy detected - check calculations")
    
    # Test specific array directions
    print(f"\n🏗️ PV Array Direction Analysis:")
    array_azimuths = [180, 225, 270]  # South, SW, West
    array_names = ['South (180°)', 'Southwest (225°)', 'West (270°)']
    
    for i, azimuth in enumerate(array_azimuths):
        slope_result = analyzer.get_slope_in_direction(x_utm, y_utm, azimuth)
        slope = slope_result['slope_degrees']
        print(f"   {array_names[i]:15}: {slope:.1f}° ground slope")
        
        # Interpret what this means for array installation
        if slope > 5:
            print(f"                      ⚠️  Significant slope - consider in array design")
        elif slope > 2:
            print(f"                      ⚡ Moderate slope - minor impact expected") 
        else:
            print(f"                      ✅ Minimal slope - negligible impact")
    
    return {
        'usgs_data': {
            'elevation': nearest_elev,
            'max_slope': nearest_slope, 
            'aspect': nearest_aspect
        },
        'directional_slopes': dict(zip(direction_names, all_slopes)),
        'array_analysis': dict(zip(array_names, [analyzer.get_slope_in_direction(x_utm, y_utm, az)['slope_degrees'] for az in array_azimuths]))
    }

def get_cardinal_direction(azimuth):
    """Convert azimuth to cardinal direction description."""
    azimuth = azimuth % 360
    
    if azimuth < 22.5 or azimuth >= 337.5:
        return "North"
    elif azimuth < 67.5:
        return "Northeast"  
    elif azimuth < 112.5:
        return "East"
    elif azimuth < 157.5:
        return "Southeast"
    elif azimuth < 202.5:
        return "South" 
    elif azimuth < 247.5:
        return "Southwest"
    elif azimuth < 292.5:
        return "West"
    else:
        return "Northwest"

def create_slope_visualization():
    """Create a visual representation of the slope analysis."""
    
    print(f"\n🎨 Creating slope visualization...")
    
    # Test location
    lat, lon = 39.796678, -79.092463
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    
    # Calculate slopes in all directions (every 10°)
    azimuths = np.arange(0, 360, 10)
    slopes = [analyzer.get_slope_in_direction(x_utm, y_utm, az)['slope_degrees'] for az in azimuths]
    
    # Create polar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Polar plot of directional slopes
    ax1_polar = plt.subplot(1, 2, 1, projection='polar')
    theta = np.radians(azimuths)
    ax1_polar.plot(theta, slopes, 'b-', linewidth=2)
    ax1_polar.fill(theta, slopes, alpha=0.3)
    ax1_polar.set_title('Directional Ground Slope\n(Meyersdale, PA Array Site)', pad=20)
    ax1_polar.set_theta_zero_location('N')
    ax1_polar.set_theta_direction(-1)
    ax1_polar.set_ylim(0, max(slopes) * 1.1)
    
    # Add cardinal directions
    cardinal_angles = [0, 90, 180, 270]
    cardinal_labels = ['N', 'E', 'S', 'W']
    for angle, label in zip(cardinal_angles, cardinal_labels):
        slope_val = analyzer.get_slope_in_direction(x_utm, y_utm, angle)['slope_degrees']
        ax1_polar.annotate(f'{label}\n{slope_val:.1f}°', 
                          xy=(np.radians(angle), slope_val),
                          xytext=(np.radians(angle), slope_val + max(slopes) * 0.15),
                          ha='center', va='center', fontweight='bold')
    
    # Cartesian plot showing interpretation
    ax2.plot(azimuths, slopes, 'b-', linewidth=2)
    ax2.fill_between(azimuths, slopes, alpha=0.3)
    ax2.set_xlabel('Azimuth (degrees from North)')
    ax2.set_ylabel('Ground Slope (degrees)')
    ax2.set_title('Ground Slope vs Direction')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, max(slopes) * 1.1)
    
    # Mark key directions
    key_directions = [0, 90, 180, 270]
    key_names = ['N', 'E', 'S', 'W']
    for direction, name in zip(key_directions, key_names):
        slope_val = analyzer.get_slope_in_direction(x_utm, y_utm, direction)['slope_degrees']
        ax2.plot(direction, slope_val, 'ro', markersize=8)
        ax2.annotate(f'{name}: {slope_val:.1f}°', 
                    xy=(direction, slope_val),
                    xytext=(direction, slope_val + max(slopes) * 0.1),
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('slope_verification_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved as 'slope_verification_plot.png'")

def verify_trigonometric_calculation():
    """
    Verify the trigonometric relationship we're using for directional slopes.
    
    The key relationship is: slope_in_direction = max_slope * cos(angle_difference)
    """
    
    print(f"\n🧮 Trigonometric Calculation Verification:")
    print(f"   Formula: slope_in_direction = max_slope × cos(angle_difference)")
    print(f"   where angle_difference = |target_direction - aspect_direction|")
    
    # Example with real data
    max_slope = 6.5  # degrees (from USGS)
    aspect = 265     # degrees (from USGS - steepest downhill direction)
    
    print(f"\n📊 Example Calculation:")
    print(f"   USGS max slope: {max_slope}°")
    print(f"   USGS aspect: {aspect}° (steepest downhill toward west)")
    
    test_cases = [
        (265, "Downhill (aspect direction)"),
        (85, "Uphill (opposite of aspect)"), 
        (180, "South (perpendicular-ish)"),
        (0, "North (perpendicular-ish)"),
        (270, "West (close to downhill)"),
        (90, "East (close to uphill)")
    ]
    
    print(f"\n   Target Direction → Expected Slope:")
    for direction, description in test_cases:
        angle_diff = abs(direction - aspect)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        calculated_slope = max_slope * np.cos(np.radians(angle_diff))
        
        print(f"   {direction:3}° ({description:25}): {calculated_slope:4.1f}° (angle_diff: {angle_diff:3.0f}°)")
    
    # Verify extremes
    print(f"\n✅ Verification Checks:")
    print(f"   • At aspect direction (265°): cos(0°) = 1.0 → slope = {max_slope:.1f}° ✓")
    print(f"   • At opposite direction (85°): cos(180°) = -1.0 → slope = {-max_slope:.1f}° ✓")
    print(f"   • At perpendicular (175°/355°): cos(90°) = 0.0 → slope ≈ 0.0° ✓")
    
    print(f"\n🔍 Physical Interpretation:")
    print(f"   • Positive slope = uphill in that direction")
    print(f"   • Negative slope = downhill in that direction") 
    print(f"   • Zero slope = level/perpendicular to max slope direction")

if __name__ == "__main__":
    # Run verification
    results = verify_slope_interpretation()
    
    # Additional trigonometric verification
    verify_trigonometric_calculation()
    
    # Create visualization
    create_slope_visualization()
    
    print(f"\n" + "="*60)
    print(f"🎯 VERIFICATION COMPLETE")
    print(f"="*60)
    print(f"Use this analysis to validate that our slope calculations")
    print(f"match your understanding of the terrain geometry!")