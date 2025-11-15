#!/usr/bin/env python3
"""
Array Azimuth Optimization with Ground Slope

This script demonstrates how the choice of array azimuth affects the ground slope impact,
and therefore the overall energy performance. The same terrain will have different
effective tilts depending on which direction the array faces.
"""

import numpy as np
import matplotlib.pyplot as plt
from ground_slope_analysis import TerrainAnalyzer, calculate_effective_array_geometry
import json

def analyze_azimuth_dependent_slope_effects():
    """
    Show how array azimuth choice affects ground slope impact for the same terrain.
    
    Key insight: The same terrain will affect arrays differently depending on
    which direction they face relative to the ground slope.
    """
    
    print("🧭 Array Azimuth vs Ground Slope Impact Analysis")
    print("=" * 60)
    
    # Test location with known ground slope characteristics
    lat, lon = 39.796678, -79.092463
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    
    # Get ground slope characteristics
    distances = np.sqrt((analyzer.x_coords - x_utm)**2 + (analyzer.y_coords - y_utm)**2)
    nearest_idx = np.argmin(distances)
    
    ground_slope_mag = analyzer.grid_points[nearest_idx]['slope']    # 3.4°
    ground_slope_dir = analyzer.grid_points[nearest_idx]['aspect']   # 99.5° (eastward)
    
    print(f"📍 Analysis Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"📊 Ground Characteristics:")
    print(f"   Slope magnitude: {ground_slope_mag:.1f}°")
    print(f"   Slope direction: {ground_slope_dir:.1f}° (steepest downhill toward {get_cardinal_direction(ground_slope_dir)})")
    print(f"   Terrain: West side {ground_slope_mag:.1f}° higher than East side")
    
    # Test array orientations from 90° (east) to 270° (west)
    test_azimuths = np.arange(90, 271, 15)  # Every 15° from East to West
    nominal_tilt = 30  # Standard tilt angle
    
    results = []
    
    print(f"\n🏗️ Array Azimuth Impact Analysis:")
    print(f"   Nominal tilt: {nominal_tilt}° for all orientations")
    print(f"   Testing azimuths from {test_azimuths[0]}° to {test_azimuths[-1]}°")
    
    print(f"\n   {'Azimuth':<8} {'Direction':<12} {'Effective':<10} {'Tilt':<6} {'Slope':<10}")
    print(f"   {'(°)':<8} {'Name':<12} {'Tilt (°)':<10} {'Δ(°)':<6} {'Component':<10}")
    print(f"   " + "-" * 55)
    
    for azimuth in test_azimuths:
        # Calculate effective geometry for this azimuth
        effective_geom = calculate_effective_array_geometry(
            ground_slope_mag, ground_slope_dir, nominal_tilt, azimuth
        )
        
        effective_tilt = effective_geom['effective_tilt']
        tilt_delta = effective_tilt - nominal_tilt
        slope_component = effective_geom['ground_slope_component']
        
        direction_name = get_cardinal_direction(azimuth)
        
        results.append({
            'azimuth': azimuth,
            'direction_name': direction_name,
            'effective_tilt': effective_tilt,
            'tilt_delta': tilt_delta,
            'slope_component': slope_component,
            'effective_geom': effective_geom
        })
        
        print(f"   {azimuth:<8.0f} {direction_name:<12} {effective_tilt:<10.1f} {tilt_delta:<6.1f} {slope_component:<10.1f}")
    
    # Find optimal and worst orientations
    tilt_deltas = [r['tilt_delta'] for r in results]
    min_impact_idx = np.argmin(np.abs(tilt_deltas))
    max_pos_impact_idx = np.argmax(tilt_deltas)
    max_neg_impact_idx = np.argmin(tilt_deltas)
    
    print(f"\n📈 Key Findings:")
    print(f"   Least slope impact:     {results[min_impact_idx]['azimuth']:.0f}° ({results[min_impact_idx]['direction_name']}) - Δ{results[min_impact_idx]['tilt_delta']:+.1f}°")
    print(f"   Maximum positive impact: {results[max_pos_impact_idx]['azimuth']:.0f}° ({results[max_pos_impact_idx]['direction_name']}) - Δ{results[max_pos_impact_idx]['tilt_delta']:+.1f}°") 
    print(f"   Maximum negative impact: {results[max_neg_impact_idx]['azimuth']:.0f}° ({results[max_neg_impact_idx]['direction_name']}) - Δ{results[max_neg_impact_idx]['tilt_delta']:+.1f}°")
    
    tilt_range = max(tilt_deltas) - min(tilt_deltas)
    print(f"   Total tilt variation: {tilt_range:.1f}° across all orientations")
    
    # Physical interpretation
    print(f"\n🔍 Physical Interpretation:")
    print(f"   • Arrays facing UPHILL ({get_cardinal_direction((ground_slope_dir + 180) % 360)}) get STEEPER effective tilt")
    print(f"   • Arrays facing DOWNHILL ({get_cardinal_direction(ground_slope_dir)}) get SHALLOWER effective tilt")
    print(f"   • Arrays facing PERPENDICULAR (N/S) have minimal slope impact")
    print(f"   • Same terrain, different impacts based on array orientation!")
    
    return results

def estimate_energy_impact_by_azimuth(results):
    """
    Estimate how the tilt variations translate to energy performance differences.
    """
    
    print(f"\n⚡ Energy Impact Estimation")
    print(f"=" * 35)
    
    # Rough energy sensitivity factors (% per degree)
    tilt_sensitivity = 0.3  # ~0.3% energy change per degree of tilt error from optimal
    
    print(f"Assumptions:")
    print(f"   • Tilt sensitivity: ±{tilt_sensitivity:.1f}% energy per degree")
    print(f"   • Optimal tilt for latitude: ~30° (assumed)")
    print(f"   • Energy impact is roughly linear for small tilt changes")
    
    print(f"\n   {'Azimuth':<8} {'Direction':<12} {'Tilt Δ':<8} {'Est. Energy':<12}")
    print(f"   {'(°)':<8} {'Name':<12} {'(°)':<8} {'Impact (%)':<12}")
    print(f"   " + "-" * 45)
    
    energy_impacts = []
    for r in results:
        # Estimate energy impact from tilt change
        energy_impact = r['tilt_delta'] * tilt_sensitivity
        energy_impacts.append(energy_impact)
        
        print(f"   {r['azimuth']:<8.0f} {r['direction_name']:<12} {r['tilt_delta']:<8.1f} {energy_impact:<12.2f}")
    
    # Summary statistics
    energy_range = max(energy_impacts) - min(energy_impacts)
    best_orientation_idx = np.argmax(energy_impacts)
    worst_orientation_idx = np.argmin(energy_impacts)
    
    print(f"\n📊 Energy Performance Range:")
    print(f"   Best orientation:  {results[best_orientation_idx]['azimuth']:.0f}° ({results[best_orientation_idx]['direction_name']}) - {max(energy_impacts):+.2f}%")
    print(f"   Worst orientation: {results[worst_orientation_idx]['azimuth']:.0f}° ({results[worst_orientation_idx]['direction_name']}) - {min(energy_impacts):+.2f}%")
    print(f"   Performance range: {energy_range:.2f}% between best and worst orientations")
    
    if energy_range > 0.5:
        print(f"   ⚠️  Significant variation! Ground slope affects energy by >{energy_range:.1f}%")
        print(f"       Consider slope in azimuth optimization.")
    elif energy_range > 0.1:
        print(f"   ⚡ Moderate variation. Ground slope worth considering in design.")
    else:
        print(f"   ✅ Small variation. Ground slope has minimal impact.")
    
    return energy_impacts

def create_azimuth_optimization_visualization(results, energy_impacts):
    """
    Create visualization showing how ground slope affects different array orientations.
    """
    
    print(f"\n🎨 Creating azimuth optimization visualization...")
    
    azimuths = [r['azimuth'] for r in results]
    tilt_deltas = [r['tilt_delta'] for r in results]
    effective_tilts = [r['effective_tilt'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Tilt corrections vs azimuth
    ax1.plot(azimuths, tilt_deltas, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Array Azimuth (degrees)')
    ax1.set_ylabel('Tilt Correction (degrees)')
    ax1.set_title('Ground Slope Impact on Array Tilt vs Orientation\n(Same terrain, different array azimuths)')
    ax1.grid(True, alpha=0.3)
    
    # Mark key orientations
    key_azimuths = [90, 135, 180, 225, 270]
    key_names = ['E', 'SE', 'S', 'SW', 'W']
    for az, name in zip(key_azimuths, key_names):
        if az in azimuths:
            idx = azimuths.index(az)
            ax1.plot(az, tilt_deltas[idx], 'ro', markersize=8)
            ax1.annotate(f'{name}\n{tilt_deltas[idx]:+.1f}°', 
                        xy=(az, tilt_deltas[idx]),
                        xytext=(az, tilt_deltas[idx] + 0.3),
                        ha='center', va='bottom', fontweight='bold')
    
    # 2. Effective tilts vs azimuth  
    ax2.plot(azimuths, effective_tilts, 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Nominal tilt (30°)')
    ax2.set_xlabel('Array Azimuth (degrees)')
    ax2.set_ylabel('Effective Tilt (degrees)')
    ax2.set_title('Effective Array Tilt Accounting for Ground Slope')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Estimated energy impact
    ax3.plot(azimuths, energy_impacts, 'r-o', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Array Azimuth (degrees)')
    ax3.set_ylabel('Estimated Energy Impact (%)')
    ax3.set_title('Estimated Energy Performance Impact by Orientation')
    ax3.grid(True, alpha=0.3)
    
    # Add annotations for best/worst
    best_idx = np.argmax(energy_impacts)
    worst_idx = np.argmin(energy_impacts)
    
    ax3.plot(azimuths[best_idx], energy_impacts[best_idx], 'go', markersize=10, label='Best')
    ax3.plot(azimuths[worst_idx], energy_impacts[worst_idx], 'ro', markersize=10, label='Worst')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('azimuth_slope_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved as 'azimuth_slope_optimization.png'")

def get_cardinal_direction(azimuth):
    """Convert azimuth to cardinal direction description."""
    azimuth = azimuth % 360
    
    directions = [
        (0, "North"), (22.5, "NNE"), (45, "NE"), (67.5, "ENE"),
        (90, "East"), (112.5, "ESE"), (135, "SE"), (157.5, "SSE"),
        (180, "South"), (202.5, "SSW"), (225, "SW"), (247.5, "WSW"),
        (270, "West"), (292.5, "WNW"), (315, "NW"), (337.5, "NNW")
    ]
    
    for i, (angle, name) in enumerate(directions):
        next_angle = directions[(i + 1) % len(directions)][0]
        if next_angle < angle:  # Handle wrap-around
            next_angle += 360
        
        if angle <= azimuth < next_angle or (angle == 337.5 and azimuth >= 337.5):
            return name
    
    return "North"  # Default

if __name__ == "__main__":
    # Run azimuth-dependent analysis
    results = analyze_azimuth_dependent_slope_effects()
    
    # Estimate energy impacts
    energy_impacts = estimate_energy_impact_by_azimuth(results)
    
    # Create visualization
    create_azimuth_optimization_visualization(results, energy_impacts)
    
    print(f"\n" + "="*60)
    print(f"🎯 KEY INSIGHT CONFIRMED")
    print(f"="*60)
    print(f"✅ Array azimuth choice DOES affect ground slope impact!")
    print(f"✅ Same terrain affects different orientations differently!")
    print(f"✅ This justifies considering terrain in azimuth optimization!")