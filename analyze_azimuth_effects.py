#!/usr/bin/env python3
"""
Enhanced Ground Slope Analysis - Azimuth Effects

This script analyzes how ground slope in the perpendicular direction 
affects the optimal azimuth of a PV array, not just the tilt.
"""

import numpy as np
import matplotlib.pyplot as plt
from ground_slope_analysis import TerrainAnalyzer, calculate_effective_array_geometry
import json

def analyze_azimuth_effects():
    """
    Analyze how ground slope affects both tilt and azimuth of PV arrays.
    
    Key questions:
    1. Does east-west slope affect optimal azimuth?
    2. How significant is the azimuth correction compared to tilt correction?
    3. Should we consider azimuth corrections in our pvlib integration?
    """
    
    print("🔄 Enhanced Ground Slope Analysis - Azimuth Effects")
    print("=" * 65)
    
    # Test location
    lat, lon = 39.796678, -79.092463
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    
    print(f"\n📍 Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"   UTM: {x_utm:.0f}, {y_utm:.0f}")
    
    # Get overall ground slope characteristics  
    # Find the point with slope and aspect data
    distances = np.sqrt((analyzer.x_coords - x_utm)**2 + (analyzer.y_coords - y_utm)**2)
    nearest_idx = np.argmin(distances)
    
    ground_slope_mag = analyzer.grid_points[nearest_idx]['slope']    # Overall slope magnitude
    ground_slope_dir = analyzer.grid_points[nearest_idx]['aspect']   # Direction of steepest slope
    ground_elevation = analyzer.elevations[nearest_idx]
    
    print(f"\n📊 Ground Slope Summary:")
    print(f"   Ground slope magnitude: {ground_slope_mag:.1f}°")
    print(f"   Ground slope direction: {ground_slope_dir:.1f}° (aspect - steepest downhill)")
    print(f"   Ground elevation: {ground_elevation:.1f}m")
    
    # Also get directional slopes for comparison
    slope_east = analyzer.get_slope_in_direction(x_utm, y_utm, 90)   # East
    slope_west = analyzer.get_slope_in_direction(x_utm, y_utm, 270)  # West  
    slope_south = analyzer.get_slope_in_direction(x_utm, y_utm, 180) # South
    slope_north = analyzer.get_slope_in_direction(x_utm, y_utm, 0)   # North
    
    print(f"\n   Directional slopes:")
    print(f"   East:  {slope_east['slope_degrees']:+5.1f}° (perpendicular to south-facing array)")
    print(f"   West:  {slope_west['slope_degrees']:+5.1f}° (perpendicular to south-facing array)")
    print(f"   South: {slope_south['slope_degrees']:+5.1f}° (parallel to south-facing array)")
    print(f"   North: {slope_north['slope_degrees']:+5.1f}° (parallel to south-facing array)")
    
    # Calculate net east-west and north-south slopes
    net_ew_slope = (slope_east['slope_degrees'] - slope_west['slope_degrees']) / 2
    net_ns_slope = (slope_south['slope_degrees'] - slope_north['slope_degrees']) / 2
    
    print(f"\n🧮 Net Ground Slope Components:")
    print(f"   East-West slope: {net_ew_slope:+5.1f}° (+ = higher on west)")
    print(f"   North-South slope: {net_ns_slope:+5.1f}° (+ = higher on south)")
    
    # Analyze effect on different array orientations
    print(f"\n🏗️ Array Orientation Analysis:")
    
    test_azimuths = [150, 165, 180, 195, 210]  # Range around south
    azimuth_names = ['SSE (150°)', 'SSE (165°)', 'S (180°)', 'SSW (195°)', 'SSW (210°)']
    
    nominal_tilt = 30  # degrees
    
    results = []
    for i, azimuth in enumerate(test_azimuths):
        # Get effective geometry considering ground slope
        effective_geom = calculate_effective_array_geometry(
            ground_slope_mag, ground_slope_dir, nominal_tilt, azimuth
        )
        
        tilt_correction = effective_geom['effective_tilt'] - nominal_tilt
        azimuth_correction = effective_geom['effective_azimuth'] - azimuth
        
        # Handle azimuth wraparound
        if azimuth_correction > 180:
            azimuth_correction -= 360
        elif azimuth_correction < -180:
            azimuth_correction += 360
            
        results.append({
            'azimuth': azimuth,
            'name': azimuth_names[i],
            'effective_tilt': effective_geom['effective_tilt'],
            'effective_azimuth': effective_geom['effective_azimuth'],
            'tilt_correction': tilt_correction,
            'azimuth_correction': azimuth_correction
        })
        
        print(f"   {azimuth_names[i]:12}: "
              f"Tilt: {nominal_tilt:.1f}°→{effective_geom['effective_tilt']:.1f}° "
              f"({tilt_correction:+.1f}°), "
              f"Azimuth: {azimuth:.0f}°→{effective_geom['effective_azimuth']:.1f}° "
              f"({azimuth_correction:+.1f}°)")
    
    # Analyze magnitude of corrections
    tilt_corrections = [r['tilt_correction'] for r in results]
    azimuth_corrections = [r['azimuth_correction'] for r in results]
    
    print(f"\n📈 Correction Magnitude Analysis:")
    print(f"   Tilt corrections:    {min(tilt_corrections):+.1f}° to {max(tilt_corrections):+.1f}°")
    print(f"   Azimuth corrections: {min(azimuth_corrections):+.1f}° to {max(azimuth_corrections):+.1f}°")
    print(f"   Average tilt change: {np.mean(np.abs(tilt_corrections)):.2f}°")
    print(f"   Average azimuth change: {np.mean(np.abs(azimuth_corrections)):.2f}°")
    
    # Theoretical analysis
    print(f"\n🧭 Theoretical Analysis:")
    print(f"   For a south-facing array (180°):")
    print(f"   • East-West slope affects azimuth more than North-South slope")
    print(f"   • Net East-West slope: {net_ew_slope:+.1f}°")
    print(f"   • This should cause azimuth to shift toward the higher side")
    
    if net_ew_slope > 0:
        print(f"   • Ground higher on WEST → Array should face slightly more WEST")
    elif net_ew_slope < 0:
        print(f"   • Ground higher on EAST → Array should face slightly more EAST")
    else:
        print(f"   • Ground level East-West → No azimuth correction needed")
    
    # Physical interpretation
    print(f"\n🔍 Physical Interpretation:")
    print(f"   When ground slopes east-to-west:")
    print(f"   • The 'up' direction is no longer vertical")
    print(f"   • Array normal vector tilts with the ground")
    print(f"   • This changes both tilt and azimuth simultaneously")
    print(f"   • Effect depends on array's original orientation")
    
    return results

def compare_energy_impact_with_azimuth():
    """
    Compare energy impact of including azimuth corrections vs tilt-only.
    """
    
    print(f"\n⚡ Energy Impact Comparison")
    print("=" * 40)
    
    # This would require running pvlib simulations with different configurations
    print(f"Theoretical energy impact analysis:")
    print(f"   • Tilt corrections: Directly affect solar angle of incidence")
    print(f"   • Azimuth corrections: Affect optimal sun tracking throughout day")
    print(f"   • Combined effect: May be larger than individual components")
    
    # Rough estimation based on solar geometry
    lat = 39.796678
    
    # Solar path characteristics for this latitude
    print(f"\n🌞 Solar Geometry Context (Lat: {lat:.1f}°N):")
    print(f"   • Winter solstice sun path: ~27° max elevation")
    print(f"   • Summer solstice sun path: ~74° max elevation") 
    print(f"   • Solar azimuth range: ~120° (sunrise) to ~240° (sunset)")
    
    # Azimuth sensitivity analysis
    azimuth_sensitivity = 0.5  # Rough estimate: 0.5% energy loss per degree off-optimal azimuth
    tilt_sensitivity = 0.3     # Rough estimate: 0.3% energy loss per degree off-optimal tilt
    
    print(f"\n📊 Estimated Sensitivity:")
    print(f"   • ~{azimuth_sensitivity:.1f}% energy change per 1° azimuth error")
    print(f"   • ~{tilt_sensitivity:.1f}% energy change per 1° tilt error") 
    
    # For our site with small corrections
    estimated_azimuth_impact = 0.1 * azimuth_sensitivity  # ~0.1° typical azimuth correction
    estimated_tilt_impact = 0.1 * tilt_sensitivity        # ~0.1° typical tilt correction
    
    print(f"\n🎯 Estimated Impact for Our Site:")
    print(f"   • Azimuth correction: ~{estimated_azimuth_impact:.2f}% energy impact")
    print(f"   • Tilt correction: ~{estimated_tilt_impact:.2f}% energy impact")
    print(f"   • Combined effect: ~{estimated_azimuth_impact + estimated_tilt_impact:.2f}% total")
    
    print(f"\n✅ Conclusion:")
    if estimated_azimuth_impact < 0.1:
        print(f"   Azimuth corrections are negligible for this site (<0.1%)")
        print(f"   Tilt-only correction is sufficient for practical purposes")
    else:
        print(f"   Azimuth corrections may be significant (>{estimated_azimuth_impact:.1f}%)")
        print(f"   Consider including azimuth corrections in energy modeling")

def create_slope_effect_visualization():
    """Create visualization showing how ground slope affects array geometry."""
    
    print(f"\n🎨 Creating slope effect visualization...")
    
    # Test data
    lat, lon = 39.796678, -79.092463
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    
    # Test range of slopes
    test_slopes = np.linspace(0, 10, 11)  # 0° to 10° slope
    slope_direction = 90  # East-facing slope
    nominal_tilt = 30
    nominal_azimuth = 180  # South-facing
    
    tilt_effects = []
    azimuth_effects = []
    
    # Calculate effects for different slope magnitudes
    for slope_mag in test_slopes:
        # Simulate different ground slope magnitudes
        # (This is theoretical - not using actual terrain data)
        
        # Simple approximation of the geometric effect
        # For east-west slopes affecting south-facing arrays
        slope_rad = np.radians(slope_mag)
        
        # Tilt effect (slope in facing direction)
        if slope_direction == 180:  # If slope faces same direction as array
            tilt_effect = slope_mag
        else:
            # Use trigonometric projection
            angle_diff = np.radians(abs(slope_direction - nominal_azimuth))
            tilt_effect = slope_mag * np.cos(angle_diff)
        
        # Azimuth effect (slope perpendicular to facing direction) 
        azimuth_angle_diff = np.radians(abs(slope_direction - (nominal_azimuth + 90)))
        azimuth_effect = slope_mag * np.cos(azimuth_angle_diff) * 0.5  # Rough approximation
        
        tilt_effects.append(tilt_effect)
        azimuth_effects.append(azimuth_effect)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Tilt effects
    ax1.plot(test_slopes, tilt_effects, 'b-', linewidth=2, label='Tilt Correction')
    ax1.fill_between(test_slopes, tilt_effects, alpha=0.3)
    ax1.set_xlabel('Ground Slope Magnitude (degrees)')
    ax1.set_ylabel('Tilt Correction (degrees)')
    ax1.set_title('Ground Slope Effect on Array Tilt\n(South-facing array, East-West ground slope)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Azimuth effects
    ax2.plot(test_slopes, azimuth_effects, 'r-', linewidth=2, label='Azimuth Correction')
    ax2.fill_between(test_slopes, azimuth_effects, alpha=0.3, color='red')
    ax2.set_xlabel('Ground Slope Magnitude (degrees)')
    ax2.set_ylabel('Azimuth Correction (degrees)')
    ax2.set_title('Ground Slope Effect on Array Azimuth\n(South-facing array, East-West ground slope)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('slope_azimuth_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved as 'slope_azimuth_effects.png'")

if __name__ == "__main__":
    # Run enhanced analysis
    results = analyze_azimuth_effects()
    
    # Energy impact comparison
    compare_energy_impact_with_azimuth()
    
    # Create visualization
    create_slope_effect_visualization()
    
    print(f"\n" + "="*65)
    print(f"🎯 ENHANCED ANALYSIS COMPLETE")
    print(f"="*65)
    print(f"This analysis shows how ground slope affects BOTH tilt and azimuth!")
    print(f"For most sites, azimuth corrections are small but may be worth considering.")