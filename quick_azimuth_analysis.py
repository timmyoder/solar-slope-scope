#!/usr/bin/env python3
"""
Simple demonstration of the azimuth vs energy curve analysis.

This script shows how to quickly analyze how array orientation affects 
energy production for any location within the terrain coverage area.
"""

from terrain_aware_optimizer import TerrainAwarePVOptimizer

def analyze_location(lat, lon, name="Test Site"):
    """
    Quick analysis of how azimuth affects energy production at a location.
    """
    
    print(f"\n🎯 Analyzing {name}")
    print(f"📍 Coordinates: {lat:.6f}°N, {lon:.6f}°W")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = TerrainAwarePVOptimizer('data/terrain_elevation_points.geojson')
    
    # Create azimuth vs energy curve
    curve_data = optimizer.plot_azimuth_energy_curve(
        lat, lon,
        tilt=30,                    # Fixed tilt angle
        azimuth_range=(160, 200),   # Focused range around south
        step_size=5,                # Base step size (for edges)
        show_terrain_benefit=True,  # Show both nominal and terrain-corrected
        adaptive_spacing=True       # 1° resolution near optimal (180°)
    )
    
    # Print key insights
    print(f"\n📊 Key Results:")
    print(f"   🏆 Optimal azimuth: {curve_data['optimal_corrected_azimuth']:.0f}°")
    print(f"   ⚡ Max energy: {curve_data['max_energy_corrected']:.0f} kWh/year")
    print(f"   🏔️ Ground slope: {curve_data['terrain_data']['slope']:.1f}° toward {curve_data['terrain_data']['aspect']:.0f}°")
    print(f"   📈 Azimuth sensitivity: {(curve_data['energy_range_corrected']/curve_data['max_energy_corrected'])*100:.1f}% range")
    print(f"   🎯 Terrain benefit: {((curve_data['max_energy_corrected']-curve_data['max_energy_nominal'])/curve_data['max_energy_nominal'])*100:.2f}%")
    
    return curve_data

def compare_terrain_effects():
    """
    Compare how different terrain conditions affect azimuth optimization.
    """
    
    print(f"\n🗺️ TERRAIN EFFECTS COMPARISON")
    print(f"=" * 60)
    
    # Test different locations within the coverage area
    test_sites = [
        (39.796678, -79.092463, "Original Site (3.4° slope)"),
        (39.800000, -79.090000, "North Location (different terrain)"),
        (39.793000, -79.095000, "South Location (different terrain)"),
    ]
    
    results = []
    
    for lat, lon, name in test_sites:
        try:
            result = analyze_location(lat, lon, name)
            results.append({
                'name': name,
                'optimal_azimuth': result['optimal_corrected_azimuth'],
                'max_energy': result['max_energy_corrected'],
                'slope': result['terrain_data']['slope'],
                'aspect': result['terrain_data']['aspect'],
                'terrain_benefit_pct': ((result['max_energy_corrected']-result['max_energy_nominal'])/result['max_energy_nominal'])*100
            })
        except Exception as e:
            print(f"   ❌ Failed to analyze {name}: {e}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n📋 COMPARISON SUMMARY:")
        print(f"   {'Location':<30} {'Optimal':<8} {'Energy':<8} {'Slope':<8} {'Benefit'}")
        print(f"   {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for r in results:
            print(f"   {r['name']:<30} {r['optimal_azimuth']:>6.0f}°  {r['max_energy']:>6.0f}   {r['slope']:>6.1f}°  {r['terrain_benefit_pct']:>+5.2f}%")

def quick_analysis_demo():
    """
    Demonstrate the quick analysis capability.
    """
    
    print(f"🚀 QUICK AZIMUTH ANALYSIS DEMO")
    print(f"=" * 60)
    
    # For any location in the coverage area, you can now run:
    #
    # 1. Quick single-location analysis
    analyze_location(39.796678, -79.092463, "Demo Site")
    
    # 2. Compare multiple locations
    # compare_terrain_effects()

if __name__ == "__main__":
    quick_analysis_demo()