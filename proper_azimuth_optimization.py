#!/usr/bin/env python3
"""
Proper Terrain-Aware Azimuth Optimization

This script properly combines solar geometry with terrain effects to find
the TRUE optimal azimuth, not just the orientation with the best slope correction.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ground_slope_analysis import TerrainAnalyzer, calculate_effective_array_geometry
import pvlib
from pvlib import location, pvsystem, modelchain

def proper_azimuth_optimization():
    """
    Properly combine solar geometry and terrain effects for azimuth optimization.
    """
    
    print("🎯 Proper Terrain-Aware Azimuth Optimization")
    print("=" * 55)
    
    # Location and terrain setup
    lat, lon = 39.796678, -79.092463
    analyzer = TerrainAnalyzer('data/terrain_elevation_points.geojson')
    x_utm, y_utm = analyzer.convert_latlon_to_utm(lat, lon)
    
    # Get ground slope characteristics
    distances = np.sqrt((analyzer.x_coords - x_utm)**2 + (analyzer.y_coords - y_utm)**2)
    nearest_idx = np.argmin(distances)
    
    ground_slope_mag = analyzer.grid_points[nearest_idx]['slope']
    ground_slope_dir = analyzer.grid_points[nearest_idx]['aspect']
    
    print(f"📍 Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"🏔️ Terrain: {ground_slope_mag:.1f}° slope toward {ground_slope_dir:.1f}° (East)")
    
    # Test azimuth range (focus around south ±60°)
    test_azimuths = np.arange(120, 241, 5)  # 120° to 240°, every 5°
    nominal_tilt = 30
    
    print(f"\n⚡ Running pvlib energy simulations for each azimuth...")
    print(f"   Testing {len(test_azimuths)} orientations from {test_azimuths[0]}° to {test_azimuths[-1]}°")
    
    # Setup pvlib location
    site = location.Location(latitude=lat, longitude=lon, tz='America/New_York')
    
    # Create simplified annual weather data
    times = pd.date_range('2023-01-01 06:00', '2023-12-31 18:00', freq='3h', tz=site.tz)
    clearsky = site.get_clearsky(times)
    weather = clearsky.copy()
    weather['temp_air'] = 25
    weather['wind_speed'] = 2
    
    results = []
    
    for azimuth in test_azimuths:
        # Get terrain-corrected geometry
        effective_geom = calculate_effective_array_geometry(
            ground_slope_mag, ground_slope_dir, nominal_tilt, azimuth
        )
        
        effective_tilt = effective_geom['effective_tilt']
        effective_azimuth = effective_geom['effective_azimuth']
        tilt_correction = effective_tilt - nominal_tilt
        
        # Run pvlib simulation with terrain corrections
        system_corrected = pvsystem.PVSystem(
            surface_tilt=effective_tilt,
            surface_azimuth=effective_azimuth,
            module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
            inverter_parameters={'pdc0': 300, 'eta_inv_nom': 0.96},
            racking_model='open_rack',
            module_type='glass_glass'
        )
        
        # Run simulation with nominal geometry (no terrain correction)
        system_nominal = pvsystem.PVSystem(
            surface_tilt=nominal_tilt,
            surface_azimuth=azimuth,
            module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
            inverter_parameters={'pdc0': 300, 'eta_inv_nom': 0.96},
            racking_model='open_rack',
            module_type='glass_glass'
        )
        
        try:
            # Corrected simulation
            mc_corrected = modelchain.ModelChain(
                system_corrected, site,
                aoi_model='no_loss',
                spectral_model='no_loss',
                temperature_model='sapm',
                losses_model='no_loss'
            )
            mc_corrected.run_model(weather)
            energy_corrected = mc_corrected.results.ac.sum() / 1000  # kWh
            
            # Nominal simulation
            mc_nominal = modelchain.ModelChain(
                system_nominal, site,
                aoi_model='no_loss',
                spectral_model='no_loss',
                temperature_model='sapm',
                losses_model='no_loss'
            )
            mc_nominal.run_model(weather)
            energy_nominal = mc_nominal.results.ac.sum() / 1000  # kWh
            
            # Calculate impacts
            terrain_benefit = energy_corrected - energy_nominal
            terrain_benefit_pct = (terrain_benefit / energy_nominal) * 100
            
            results.append({
                'azimuth': azimuth,
                'energy_nominal': energy_nominal,
                'energy_corrected': energy_corrected,
                'terrain_benefit_kwh': terrain_benefit,
                'terrain_benefit_pct': terrain_benefit_pct,
                'effective_tilt': effective_tilt,
                'tilt_correction': tilt_correction
            })
            
        except Exception as e:
            print(f"   ⚠️ Simulation failed for azimuth {azimuth}°: {e}")
    
    if not results:
        print("❌ No successful simulations")
        return None
    
    # Find optimal orientations
    energies_nominal = [r['energy_nominal'] for r in results]
    energies_corrected = [r['energy_corrected'] for r in results]
    azimuths = [r['azimuth'] for r in results]
    
    # Optimal without terrain consideration
    optimal_nominal_idx = np.argmax(energies_nominal)
    optimal_nominal_azimuth = azimuths[optimal_nominal_idx]
    optimal_nominal_energy = energies_nominal[optimal_nominal_idx]
    
    # Optimal with terrain consideration  
    optimal_corrected_idx = np.argmax(energies_corrected)
    optimal_corrected_azimuth = azimuths[optimal_corrected_idx]
    optimal_corrected_energy = energies_corrected[optimal_corrected_idx]
    
    print(f"\n📊 Optimization Results:")
    print(f"   🏠 Optimal azimuth (NO terrain):  {optimal_nominal_azimuth:3.0f}° - {optimal_nominal_energy:.0f} kWh/year")
    print(f"   🏔️ Optimal azimuth (WITH terrain): {optimal_corrected_azimuth:3.0f}° - {optimal_corrected_energy:.0f} kWh/year")
    
    azimuth_shift = optimal_corrected_azimuth - optimal_nominal_azimuth
    energy_gain = optimal_corrected_energy - optimal_nominal_energy
    energy_gain_pct = (energy_gain / optimal_nominal_energy) * 100
    
    print(f"\n🎯 Terrain Impact on Optimization:")
    print(f"   Azimuth shift: {azimuth_shift:+.0f}° (terrain pulls optimum toward {azimuth_shift:+.0f}°)")
    print(f"   Energy benefit: +{energy_gain:.1f} kWh/year ({energy_gain_pct:+.2f}%)")
    
    if abs(azimuth_shift) > 2:
        print(f"   ⚠️ Significant shift! Terrain changes optimal azimuth by {abs(azimuth_shift):.0f}°")
    elif abs(azimuth_shift) > 0.5:
        print(f"   📍 Moderate shift. Terrain fine-tunes optimal azimuth.")
    else:
        print(f"   ✅ Small shift. Solar geometry dominates, terrain has minor impact.")
    
    # Show comparison at traditional 180° south
    south_idx = None
    for i, az in enumerate(azimuths):
        if az == 180:
            south_idx = i
            break
    
    if south_idx is not None:
        south_nominal = energies_nominal[south_idx]
        south_corrected = energies_corrected[south_idx]
        
        print(f"\n🧭 Traditional South (180°) Comparison:")
        print(f"   South nominal: {south_nominal:.0f} kWh/year")
        print(f"   South corrected: {south_corrected:.0f} kWh/year") 
        print(f"   vs Optimal nominal: {((south_nominal/optimal_nominal_energy)-1)*100:+.1f}%")
        print(f"   vs Optimal corrected: {((south_corrected/optimal_corrected_energy)-1)*100:+.1f}%")
    
    return results

def create_proper_optimization_plots(results):
    """Create visualization showing proper solar + terrain optimization."""
    
    if not results:
        return
    
    print(f"\n🎨 Creating proper optimization visualization...")
    
    azimuths = [r['azimuth'] for r in results]
    energies_nominal = [r['energy_nominal'] for r in results]
    energies_corrected = [r['energy_corrected'] for r in results]
    terrain_benefits = [r['terrain_benefit_pct'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Energy vs azimuth (both nominal and corrected)
    ax1.plot(azimuths, energies_nominal, 'b-', linewidth=2, label='Solar geometry only', alpha=0.7)
    ax1.plot(azimuths, energies_corrected, 'r-', linewidth=2, label='Solar + terrain correction')
    
    # Mark optima
    optimal_nominal_idx = np.argmax(energies_nominal)
    optimal_corrected_idx = np.argmax(energies_corrected)
    
    ax1.plot(azimuths[optimal_nominal_idx], energies_nominal[optimal_nominal_idx], 
             'bo', markersize=10, label=f'Optimal (no terrain): {azimuths[optimal_nominal_idx]:.0f}°')
    ax1.plot(azimuths[optimal_corrected_idx], energies_corrected[optimal_corrected_idx], 
             'ro', markersize=10, label=f'Optimal (with terrain): {azimuths[optimal_corrected_idx]:.0f}°')
    
    ax1.set_xlabel('Array Azimuth (degrees)')
    ax1.set_ylabel('Annual Energy (kWh)')
    ax1.set_title('Annual Energy vs Array Azimuth\n(Proper solar geometry + terrain optimization)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Terrain benefit percentage
    ax2.plot(azimuths, terrain_benefits, 'g-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Array Azimuth (degrees)')
    ax2.set_ylabel('Terrain Benefit (%)')
    ax2.set_title('Energy Benefit from Terrain Corrections')
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy comparison
    energy_diff = np.array(energies_corrected) - np.array(energies_nominal)
    ax3.plot(azimuths, energy_diff, 'purple', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Array Azimuth (degrees)')
    ax3.set_ylabel('Energy Gain from Terrain (kWh)')
    ax3.set_title('Absolute Energy Gain from Terrain Corrections')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('proper_terrain_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved as 'proper_terrain_optimization.png'")

if __name__ == "__main__":
    # Run proper optimization
    results = proper_azimuth_optimization()
    
    # Create visualization
    if results:
        create_proper_optimization_plots(results)
        
        print(f"\n" + "="*55)
        print(f"🎯 PROPER OPTIMIZATION COMPLETE")
        print(f"="*55)
        print(f"This shows the REAL optimal azimuth considering both")
        print(f"solar geometry AND terrain effects together!")
        print(f"The previous 105° result ignored solar geometry entirely!")
    else:
        print("❌ Analysis failed - check pvlib configuration")