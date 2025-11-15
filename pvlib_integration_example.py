#!/usr/bin/env python3
"""
Complete Ground Slope + pvlib Integration Example
================================================

This script demonstrates how to use our terrain slope analysis results
with pvlib for complete energy modeling that accounts for ground slope.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Import our custom ground slope analysis
from ground_slope_analysis import analyze_array_location, TerrainAnalyzer

# Import pvlib for solar calculations
import pvlib
from pvlib import location, pvsystem, modelchain, clearsky

# Constants
LAT_CENTER, LON_CENTER = 39.796678, -79.092463  # Meyersdale, PA
TERRAIN_FILE = 'data/terrain_elevation_points.geojson'


def create_pvlib_system_with_slope_correction(lat, lon, terrain_file, 
                                            nominal_tilt=30, nominal_azimuth=180):
    """
    Create pvlib PVSystem object with ground slope correction applied.
    
    Parameters:
    -----------
    lat, lon : float
        Array location coordinates
    terrain_file : str
        Path to terrain data file
    nominal_tilt : float
        Desired array tilt relative to ground (degrees)
    nominal_azimuth : float
        Array azimuth direction (degrees, 0° = North, 180° = South)
        
    Returns:
    --------
    dict : {
        'site': pvlib.location.Location object,
        'system': pvlib.pvsystem.PVSystem object (slope-corrected),
        'slope_analysis': dict with slope analysis results,
        'correction_applied': bool
    }
    """
    
    print(f"🔧 Creating pvlib system with ground slope correction...")
    print(f"   Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"   Nominal configuration: {nominal_tilt}° tilt, {nominal_azimuth}° azimuth")
    
    # Step 1: Analyze ground slope at array location
    slope_results = analyze_array_location(
        terrain_file, lat, lon, nominal_azimuth, nominal_tilt
    )
    
    # Step 2: Extract effective geometry (corrected for ground slope)
    effective_geometry = slope_results['effective_geometry']
    
    # Step 3: Create pvlib Location object
    site = location.Location(
        latitude=lat,
        longitude=lon,
        tz='America/New_York',  # Eastern timezone for PA
        altitude=slope_results['ground_slope']['elevation']  # Use actual ground elevation
    )
    
    # Step 4: Create pvlib PVSystem with CORRECTED parameters
    # These are the key parameters that incorporate ground slope:
    effective_tilt = effective_geometry['effective_tilt']
    effective_azimuth = effective_geometry['effective_azimuth']
    
    # Create pvlib PVSystem with slope-corrected parameters (with required SAPM attributes)
    system = pvsystem.PVSystem(
        surface_tilt=effective_tilt,
        surface_azimuth=effective_azimuth,
        module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 300, 'eta_inv_nom': 0.96},
        racking_model='open_rack',  # Required for SAPM temperature model
        module_type='glass_glass'   # Required for SAPM temperature model
    )
    
    # Step 5: Calculate correction metrics
    tilt_correction = effective_tilt - nominal_tilt
    significant_correction = abs(tilt_correction) > 1.0
    
    print(f"✅ pvlib system created with slope correction:")
    print(f"   Effective tilt: {effective_tilt:.1f}° (nominal: {nominal_tilt:.1f}°, Δ{tilt_correction:+.1f}°)")
    print(f"   Effective azimuth: {effective_azimuth:.1f}°")
    print(f"   Ground elevation: {slope_results['ground_slope']['elevation']:.1f}m")
    print(f"   Slope correction: {'SIGNIFICANT' if significant_correction else 'minimal'}")
    
    return {
        'site': site,
        'system': system,
        'slope_analysis': slope_results,
        'correction_applied': significant_correction
    }


def run_annual_energy_simulation(pvsystem_obj, site_location):
    """
    Run a simplified annual energy simulation.
    
    This creates clear-sky weather data and runs the simulation.
    """
    try:
        print("\n⚡ Running annual energy simulation for 2023...")
        
        # Create simplified time range
        times = pd.date_range(
            '2023-01-01 06:00:00-05:00',
            '2023-12-31 18:00:00-05:00',
            freq='h'
        )
        print(f"   Simulation period: {times[0]} to {times[-1]}")
        print(f"   Time steps: {len(times):,} hourly intervals")
        
        # Create simple clear-sky weather data 
        clearsky = site_location.get_clearsky(times)
        
        # Add temperature and wind speed (required for modeling)
        weather = clearsky.copy()
        weather['temp_air'] = 25  # Constant 25°C air temperature
        weather['wind_speed'] = 2  # Constant 2 m/s wind speed
        
        # Simple ModelChain with all models explicitly set to avoid inference
        mc = modelchain.ModelChain(
            pvsystem_obj, 
            site_location,
            aoi_model='no_loss',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
        )
        
        # Run the simulation
        mc.run_model(weather)
        
        if mc.results.ac is None or mc.results.ac.empty:
            raise ValueError("ModelChain simulation produced no results")
        
        # Calculate annual energy (kWh)
        annual_ac = mc.results.ac.sum() / 1000  # Convert Wh to kWh
        
        return {
            'annual_energy_kwh': annual_ac,
            'weather': weather,
            'results': mc.results,
            'model_chain': mc
        }
        
    except Exception as e:
        print(f"❌ Error during energy simulation: {e}")
        return None


def compare_slope_vs_no_slope(lat, lon, terrain_file, nominal_tilt=30, nominal_azimuth=180):
    """
    Compare energy production with and without ground slope correction.
    
    This demonstrates the impact of considering terrain slope in PV modeling.
    """
    
    print(f"\n📊 Comparing energy production: slope-corrected vs. nominal configuration")
    print(f"=" * 80)
    
    # System WITH slope correction
    slope_system = create_pvlib_system_with_slope_correction(
        lat, lon, terrain_file, nominal_tilt, nominal_azimuth
    )
    
    # System WITHOUT slope correction (nominal configuration)
    nominal_site = location.Location(
        latitude=lat, longitude=lon, tz='America/New_York'
    )
    
    nominal_pvsystem = pvsystem.PVSystem(
        surface_tilt=nominal_tilt,
        surface_azimuth=nominal_azimuth,
        module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 300, 'eta_inv_nom': 0.96},
        racking_model='open_rack',  # Required for SAPM temperature model
        module_type='glass_glass'   # Required for SAPM temperature model
    )
    
    print(f"\n🏔️  System WITH slope correction:")
    print(f"   Tilt: {slope_system['slope_analysis']['effective_geometry']['effective_tilt']:.1f}°")
    print(f"   Azimuth: {slope_system['slope_analysis']['effective_geometry']['effective_azimuth']:.1f}°")
    
    print(f"\n🏠  System WITHOUT slope correction (nominal):")
    print(f"   Tilt: {nominal_tilt:.1f}°") 
    print(f"   Azimuth: {nominal_azimuth:.1f}°")
    
    # Run simulations for both systems
    try:
        slope_results = run_annual_energy_simulation(slope_system['system'], slope_system['site'])
        nominal_results = run_annual_energy_simulation(nominal_pvsystem, nominal_site)
    except Exception as e:
        print(f"❌ Error during energy simulation: {e}")
        print(f"   This may be due to pvlib version compatibility or missing weather data.")
        return None
    
    # Calculate differences
    energy_diff = slope_results['annual_energy_kwh'] - nominal_results['annual_energy_kwh']
    percent_diff = (energy_diff / nominal_results['annual_energy_kwh']) * 100
    
    print(f"\n📈 COMPARISON RESULTS:")
    print(f"   Slope-corrected: {slope_results['annual_energy_kwh']:.1f} kWh/year")
    print(f"   Nominal config:  {nominal_results['annual_energy_kwh']:.1f} kWh/year")
    print(f"   Difference: {energy_diff:+.1f} kWh/year ({percent_diff:+.1f}%)")
    
    if abs(percent_diff) > 1.0:
        print(f"   ⚠️  Significant difference! Ground slope correction is important for this site.")
    else:
        print(f"   ✅ Small difference - ground slope has minimal impact at this location.")
    
    return {
        'slope_corrected': slope_results,
        'nominal': nominal_results,
        'energy_difference_kwh': energy_diff,
        'percent_difference': percent_diff,
        'slope_analysis': slope_system['slope_analysis']
    }


def main():
    """Main function demonstrating complete pvlib integration workflow."""
    
    print("🌞 pvlib Integration with Ground Slope Correction")
    print("=" * 60)
    print("This example shows how to use terrain slope analysis results")
    print("with pvlib for complete energy modeling.")
    print()
    
    # Check pvlib version
    print(f"📋 pvlib version: {pvlib.__version__}")
    
    # Run comparison analysis
    try:
        comparison = compare_slope_vs_no_slope(
            LAT_CENTER, LON_CENTER, TERRAIN_FILE,
            nominal_tilt=30,      # Standard tilt for this latitude
            nominal_azimuth=180   # South-facing
        )
        
        if comparison is None:
            print(f"\n❌ Comparison failed - cannot create plots")
            return
        
        # Create visualization (disabled for now - monthly data not implemented)
        print(f"   Annual energy difference: {comparison['energy_difference_kwh']:+.1f} kWh ({comparison['percent_difference']:+.1f}%)")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"INTEGRATION COMPLETE!")
        print(f"="*60)
        print(f"✅ Ground slope analysis integrated with pvlib energy modeling")
        print(f"✅ Terrain slope correction: {comparison['slope_analysis']['effective_geometry']['slope_correction']:+.1f}° tilt adjustment")
        print(f"✅ Annual energy impact: {comparison['percent_difference']:+.1f}% ({comparison['energy_difference_kwh']:+.1f} kWh)")
        
        print(f"\n🔑 Key Integration Points:")
        print(f"   • effective_tilt → pvlib surface_tilt parameter") 
        print(f"   • effective_azimuth → pvlib surface_azimuth parameter")
        print(f"   • Ground elevation → pvlib site.altitude")
        print(f"   • Complete ModelChain simulation with slope-corrected geometry")
        
        if abs(comparison['percent_difference']) > 2.0:
            print(f"\n⚠️  IMPORTANT: >2% energy difference detected!")
            print(f"   Ground slope correction is significant for this site.")
            print(f"   Always include terrain analysis in energy predictions.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find terrain data file: {TERRAIN_FILE}")
        print(f"   Please ensure terrain data is available for analysis.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Check that all dependencies (pvlib, scipy, etc.) are installed.")


if __name__ == "__main__":
    main()