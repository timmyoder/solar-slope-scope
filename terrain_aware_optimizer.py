#!/usr/bin/env python3
"""
Terrain-Aware PV Optimization Tool

This tool can analyze any arbitrary point within our terrain dataset and calculate
the optimal array orientation considering both solar geometry and ground slope effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ground_slope_analysis import TerrainAnalyzer, calculate_effective_array_geometry
import pvlib
from pvlib import location, pvsystem, modelchain
import json
from scipy.interpolate import UnivariateSpline

class TerrainAwarePVOptimizer:
    """
    Complete tool for optimizing PV array orientation considering terrain effects.
    """
    
    def __init__(self, terrain_file, timezone='America/New_York'):
        """
        Initialize the optimizer with terrain data.
        
        Parameters:
        -----------
        terrain_file : str
            Path to terrain GeoJSON file
        timezone : str
            Timezone for the location
        """
        print("🌄 Initializing Terrain-Aware PV Optimizer")
        print("=" * 50)
        
        self.analyzer = TerrainAnalyzer(terrain_file)
        self.timezone = timezone
        
        # Cache weather data for efficiency
        self._weather_cache = {}
        
        print(f"✅ Optimizer ready for terrain analysis")
        print(f"   Coverage area: {self.analyzer.bounds}")
        print(f"   Grid points: {len(self.analyzer.grid_points):,}")
    
    def optimize_location(self, lat, lon, tilt_range=None, azimuth_range=None, 
                         step_size=5, detailed_output=True):
        """
        Find optimal tilt and azimuth for a specific location considering terrain.
        
        Parameters:
        -----------
        lat, lon : float
            Location coordinates
        tilt_range : tuple, optional
            (min_tilt, max_tilt) in degrees. Default: (15, 45)
        azimuth_range : tuple, optional
            (min_azimuth, max_azimuth) in degrees. Default: (135, 225) 
        step_size : float
            Step size for optimization grid in degrees
        detailed_output : bool
            Whether to print detailed analysis
            
        Returns:
        --------
        dict : Optimization results
        """
        
        if detailed_output:
            print(f"\n🎯 Optimizing PV Array for Location")
            print(f"=" * 40)
            print(f"📍 Coordinates: {lat:.6f}°N, {lon:.6f}°W")
        
        # Check if location is within terrain coverage
        x_utm, y_utm = self.analyzer.convert_latlon_to_utm(lat, lon)
        
        if not (self.analyzer.bounds['x_min'] <= x_utm <= self.analyzer.bounds['x_max'] and
                self.analyzer.bounds['y_min'] <= y_utm <= self.analyzer.bounds['y_max']):
            raise ValueError(f"Location outside terrain coverage area")
        
        # Get terrain characteristics
        terrain_data = self._get_terrain_characteristics(x_utm, y_utm)
        
        if detailed_output:
            print(f"🏔️ Terrain Analysis:")
            print(f"   Elevation: {terrain_data['elevation']:.1f}m")
            print(f"   Ground slope: {terrain_data['slope']:.1f}° toward {terrain_data['aspect']:.1f}°")
            print(f"   Slope direction: {self._get_cardinal_direction(terrain_data['aspect'])}")
        
        # Set optimization ranges
        if tilt_range is None:
            tilt_range = (15, 45)
        if azimuth_range is None:
            azimuth_range = (120, 240)  # Match previous analysis range
        
        # Create optimization grid
        tilts = np.arange(tilt_range[0], tilt_range[1] + step_size, step_size)
        azimuths = np.arange(azimuth_range[0], azimuth_range[1] + step_size, step_size)
        
        if detailed_output:
            print(f"\n⚡ Running optimization...")
            print(f"   Tilt range: {tilt_range[0]}° to {tilt_range[1]}°")
            print(f"   Azimuth range: {azimuth_range[0]}° to {azimuth_range[1]}°")
            print(f"   Grid size: {len(tilts)} × {len(azimuths)} = {len(tilts) * len(azimuths)} configurations")
        
        # Get weather data for this location
        weather = self._get_weather_data(lat, lon)
        site = location.Location(latitude=lat, longitude=lon, tz=self.timezone)
        
        # Optimization results storage
        results = []
        best_energy = 0
        best_config = None
        
        # Test all tilt/azimuth combinations
        for tilt in tilts:
            for azimuth in azimuths:
                try:
                    # Calculate terrain-corrected geometry
                    effective_geom = calculate_effective_array_geometry(
                        terrain_data['slope'], terrain_data['aspect'], tilt, azimuth
                    )
                    
                    # Run energy simulation
                    energy = self._simulate_energy(
                        site, weather, effective_geom['effective_tilt'], 
                        effective_geom['effective_azimuth']
                    )
                    
                    result = {
                        'nominal_tilt': tilt,
                        'nominal_azimuth': azimuth,
                        'effective_tilt': effective_geom['effective_tilt'],
                        'effective_azimuth': effective_geom['effective_azimuth'],
                        'tilt_correction': effective_geom['effective_tilt'] - tilt,
                        'azimuth_correction': effective_geom['effective_azimuth'] - azimuth,
                        'annual_energy_kwh': energy,
                        'terrain_benefit': effective_geom['slope_correction']
                    }
                    
                    results.append(result)
                    
                    # Track best configuration
                    if energy > best_energy:
                        best_energy = energy
                        best_config = result.copy()
                        
                except Exception as e:
                    if detailed_output and len(results) == 0:
                        print(f"   ⚠️ Simulation failed for tilt={tilt}°, azimuth={azimuth}°: {e}")
                    continue
        
        if not results:
            raise RuntimeError("No successful optimizations - check pvlib configuration")
        
        # Analysis of results
        optimization_summary = self._analyze_optimization_results(
            results, best_config, terrain_data, detailed_output
        )
        
        return optimization_summary
    
    def plot_azimuth_energy_curve(self, lat, lon, tilt=30, azimuth_range=(160, 200), 
                                 step_size=5, show_terrain_benefit=True, adaptive_spacing=True):
        """
        Create azimuth vs energy curve showing how orientation affects energy production.
        
        Parameters:
        -----------
        lat, lon : float
            Location coordinates
        tilt : float
            Fixed tilt angle for analysis
        azimuth_range : tuple
            (min_azimuth, max_azimuth) in degrees
        step_size : float
            Base azimuth step size in degrees (used for adaptive spacing)
        show_terrain_benefit : bool
            Whether to show both nominal and terrain-corrected curves
        adaptive_spacing : bool
            Whether to use finer spacing near 180° (optimal region)
            
        Returns:
        --------
        dict : Curve data and analysis results
        """
        
        print(f"\n📈 Creating Azimuth vs Energy Curve")
        print(f"=" * 40)
        print(f"📍 Location: {lat:.6f}°N, {lon:.6f}°W")
        print(f"📐 Fixed tilt: {tilt}°")
        print(f"🧭 Azimuth range: {azimuth_range[0]}° to {azimuth_range[1]}°")
        print(f"📐 Adaptive spacing: {adaptive_spacing}")
        
        # Get terrain characteristics
        x_utm, y_utm = self.analyzer.convert_latlon_to_utm(lat, lon)
        terrain_data = self._get_terrain_characteristics(x_utm, y_utm)
        
        print(f"🏔️ Ground slope: {terrain_data['slope']:.1f}° toward {terrain_data['aspect']:.1f}° ({self._get_cardinal_direction(terrain_data['aspect'])})")
        
        # Create azimuth array with adaptive spacing
        if adaptive_spacing:
            # Fine resolution (1°) around south (170-190°)
            # Coarse resolution at the edges (160-170° and 190-200°)
            azimuths = []
            
            # Coarse spacing for edges (160-170°)
            if azimuth_range[0] < 170:
                azimuths.extend(range(azimuth_range[0], 170, step_size))
            
            # Fine spacing around optimal (170-190°) - 1° resolution
            fine_start = max(170, azimuth_range[0])
            fine_end = min(190, azimuth_range[1])
            azimuths.extend(range(fine_start, fine_end + 1, 1))
            
            # Coarse spacing for edges (190-200°)
            if azimuth_range[1] > 190:
                azimuths.extend(range(195, azimuth_range[1] + step_size, step_size))
            
            # Remove duplicates and sort
            azimuths = sorted(list(set(azimuths)))
            azimuths = np.array(azimuths)
            
            print(f"📊 Adaptive grid: {len(azimuths)} points (1° near 180°, {step_size}° at edges)")
        else:
            # Standard even spacing
            azimuths = np.arange(azimuth_range[0], azimuth_range[1] + step_size, step_size)
            print(f"📊 Even spacing: {len(azimuths)} points at {step_size}° intervals")
        
        # Get weather data
        weather = self._get_weather_data(lat, lon)
        site = location.Location(latitude=lat, longitude=lon, tz=self.timezone)
        
        # Calculate energy for each azimuth
        energies_nominal = []
        energies_corrected = []
        tilt_corrections = []
        azimuth_corrections = []
        
        print(f"⚡ Calculating energy for {len(azimuths)} orientations...")
        
        for azimuth in azimuths:
            try:
                # Calculate terrain-corrected geometry
                effective_geom = calculate_effective_array_geometry(
                    terrain_data['slope'], terrain_data['aspect'], tilt, azimuth
                )
                
                # Nominal energy (flat ground)
                energy_nominal = self._simulate_energy(
                    site, weather, tilt, azimuth
                )
                
                # Terrain-corrected energy
                energy_corrected = self._simulate_energy(
                    site, weather, effective_geom['effective_tilt'], 
                    effective_geom['effective_azimuth']
                )
                
                energies_nominal.append(energy_nominal)
                energies_corrected.append(energy_corrected)
                tilt_corrections.append(effective_geom['effective_tilt'] - tilt)
                azimuth_corrections.append(effective_geom['effective_azimuth'] - azimuth)
                
            except Exception as e:
                print(f"   ⚠️ Failed at azimuth {azimuth}°: {e}")
                energies_nominal.append(np.nan)
                energies_corrected.append(np.nan)
                tilt_corrections.append(np.nan)
                azimuth_corrections.append(np.nan)
        
        # Convert to arrays
        energies_nominal = np.array(energies_nominal)
        energies_corrected = np.array(energies_corrected)
        tilt_corrections = np.array(tilt_corrections)
        azimuth_corrections = np.array(azimuth_corrections)
        
        # Find optimal points
        valid_mask = ~np.isnan(energies_nominal)
        if not np.any(valid_mask):
            raise RuntimeError("No valid energy calculations")
        
        optimal_nominal_idx = np.nanargmax(energies_nominal)
        optimal_corrected_idx = np.nanargmax(energies_corrected)
        
        optimal_nominal_azimuth = azimuths[optimal_nominal_idx]
        optimal_corrected_azimuth = azimuths[optimal_corrected_idx]
        
        max_energy_nominal = energies_nominal[optimal_nominal_idx]
        max_energy_corrected = energies_corrected[optimal_corrected_idx]
        
        # Create main visualization - focused 2-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main energy curve plot
        # Plot data points
        valid_azimuths = azimuths[valid_mask]
        valid_nominal = energies_nominal[valid_mask]
        valid_corrected = energies_corrected[valid_mask]
        
        ax1.scatter(valid_azimuths, valid_nominal, alpha=0.6, color='lightblue', 
                   label='Nominal (flat ground)', s=30)
        
        if show_terrain_benefit:
            ax1.scatter(valid_azimuths, valid_corrected, alpha=0.6, color='orange', 
                       label='Terrain-corrected', s=30)
        
        # Add smooth spline curves
        if len(valid_azimuths) > 3:  # Need enough points for spline
            azimuth_smooth = np.linspace(valid_azimuths.min(), valid_azimuths.max(), 200)
            
            try:
                spline_nominal = UnivariateSpline(valid_azimuths, valid_nominal, s=len(valid_azimuths)*0.1)
                energy_smooth_nominal = spline_nominal(azimuth_smooth)
                ax1.plot(azimuth_smooth, energy_smooth_nominal, 'b-', linewidth=2.5, 
                        label='Nominal curve', alpha=0.9)
                
                if show_terrain_benefit:
                    spline_corrected = UnivariateSpline(valid_azimuths, valid_corrected, s=len(valid_azimuths)*0.1)
                    energy_smooth_corrected = spline_corrected(azimuth_smooth)
                    ax1.plot(azimuth_smooth, energy_smooth_corrected, 'r-', linewidth=2.5, 
                            label='Terrain-corrected curve', alpha=0.9)
            except:
                print("   ⚠️ Spline fitting failed, showing points only")
        
        # Mark optimal points
        ax1.axvline(optimal_nominal_azimuth, color='blue', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Optimal nominal: {optimal_nominal_azimuth:.0f}°')
        
        if show_terrain_benefit:
            ax1.axvline(optimal_corrected_azimuth, color='red', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Optimal corrected: {optimal_corrected_azimuth:.0f}°')
        
        # Mark traditional south
        ax1.axvline(180, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='South (180°)')
        
        ax1.set_xlabel('Array Azimuth (degrees)', fontsize=12)
        ax1.set_ylabel('Annual Energy (kWh)', fontsize=12)
        ax1.set_title(f'Energy vs Azimuth (Tilt = {tilt}°)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Enhanced summary statistics panel
        ax2.axis('off')
        
        # Calculate enhanced summary stats
        energy_range_nominal = np.nanmax(energies_nominal) - np.nanmin(energies_nominal)
        energy_range_corrected = np.nanmax(energies_corrected) - np.nanmin(energies_corrected) if show_terrain_benefit else 0
        
        south_idx = np.argmin(np.abs(azimuths - 180))
        south_energy_nominal = energies_nominal[south_idx] if not np.isnan(energies_nominal[south_idx]) else 0
        south_energy_corrected = energies_corrected[south_idx] if show_terrain_benefit and not np.isnan(energies_corrected[south_idx]) else 0
        
        # Performance comparison with south
        optimal_vs_south_nominal = ((max_energy_nominal - south_energy_nominal) / south_energy_nominal) * 100
        optimal_vs_south_corrected = ((max_energy_corrected - south_energy_corrected) / south_energy_corrected) * 100 if show_terrain_benefit else 0
        
        summary_text = f"""OPTIMIZATION RESULTS

Location: {lat:.6f}N, {lon:.6f}W
Ground Slope: {terrain_data['slope']:.1f} toward {terrain_data['aspect']:.1f} ({self._get_cardinal_direction(terrain_data['aspect'])})

OPTIMAL CONFIGURATION:
   Nominal optimum:     {optimal_nominal_azimuth:3.0f} -> {max_energy_nominal:6.0f} kWh/year
   Terrain-corrected:   {optimal_corrected_azimuth:3.0f} -> {max_energy_corrected:6.0f} kWh/year

TRADITIONAL SOUTH (180):
   Nominal:             {south_energy_nominal:6.0f} kWh/year
   Terrain-corrected:   {south_energy_corrected:6.0f} kWh/year

PERFORMANCE COMPARISON:
   Optimal vs South (nominal):     {optimal_vs_south_nominal:+5.2f}%
   Optimal vs South (terrain):     {optimal_vs_south_corrected:+5.2f}%
   Terrain benefit at optimum:     {((max_energy_corrected-max_energy_nominal)/max_energy_nominal)*100:+5.2f}%

SENSITIVITY ANALYSIS:
   Energy range (nominal):         {energy_range_nominal:5.1f} kWh ({(energy_range_nominal/max_energy_nominal)*100:4.1f}%)
   Energy range (terrain):         {energy_range_corrected:5.1f} kWh ({(energy_range_corrected/max_energy_corrected)*100:4.1f}%)
   Max tilt correction:            +/-{np.nanmax(np.abs(tilt_corrections)):4.1f}

KEY INSIGHTS:
   * Solar geometry dominates optimization
   * Terrain provides {((max_energy_corrected-max_energy_nominal)/max_energy_nominal)*100:+.2f}% fine-tuning
   * Azimuth choice affects {(energy_range_nominal/max_energy_nominal)*100:.1f}% energy range
   * {len(azimuths)} orientations tested with adaptive resolution"""
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.1))
        
        plt.suptitle(f'Terrain-Aware PV Azimuth Optimization\nLocation: {lat:.6f}°N, {lon:.6f}°W', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save main plot
        main_filename = f'azimuth_optimization_{lat:.3f}_{lon:.3f}.png'
        plt.savefig(main_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Main analysis saved as '{main_filename}'")
        
        # Create optional terrain details plot
        if show_terrain_benefit:
            self._create_terrain_details_plot(valid_azimuths, valid_nominal, valid_corrected, 
                                             tilt_corrections[valid_mask], azimuth_corrections[valid_mask],
                                             lat, lon, terrain_data)
        
        # Return analysis data
        return {
            'azimuths': azimuths,
            'energies_nominal': energies_nominal,
            'energies_corrected': energies_corrected,
            'tilt_corrections': tilt_corrections,
            'azimuth_corrections': azimuth_corrections,
            'optimal_nominal_azimuth': optimal_nominal_azimuth,
            'optimal_corrected_azimuth': optimal_corrected_azimuth,
            'max_energy_nominal': max_energy_nominal,
            'max_energy_corrected': max_energy_corrected,
            'terrain_data': terrain_data,
            'energy_range_nominal': energy_range_nominal,
            'energy_range_corrected': energy_range_corrected if show_terrain_benefit else 0
        }
    
    def _create_terrain_details_plot(self, valid_azimuths, valid_nominal, valid_corrected,
                                   tilt_corrections, azimuth_corrections, lat, lon, terrain_data):
        """Create separate plot showing terrain correction details."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Terrain benefit plot
        ax1 = axes[0, 0]
        benefit_kwh = valid_corrected - valid_nominal
        benefit_pct = (benefit_kwh / valid_nominal) * 100
        
        ax1.scatter(valid_azimuths, benefit_pct, alpha=0.7, color='green', s=50)
        ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Array Azimuth (degrees)', fontsize=11)
        ax1.set_ylabel('Terrain Benefit (%)', fontsize=11)
        ax1.set_title('Terrain Energy Benefit vs Azimuth', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line if enough points
        if len(valid_azimuths) > 5:
            try:
                spline_benefit = UnivariateSpline(valid_azimuths, benefit_pct, s=len(valid_azimuths)*0.1)
                azimuth_smooth = np.linspace(valid_azimuths.min(), valid_azimuths.max(), 200)
                benefit_smooth = spline_benefit(azimuth_smooth)
                ax1.plot(azimuth_smooth, benefit_smooth, 'g-', alpha=0.6, linewidth=2)
            except:
                pass
        
        # Tilt corrections plot
        ax2 = axes[0, 1]
        ax2.scatter(valid_azimuths, tilt_corrections, alpha=0.7, color='purple', s=50)
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Array Azimuth (degrees)', fontsize=11)
        ax2.set_ylabel('Tilt Correction (degrees)', fontsize=11)
        ax2.set_title('Terrain Tilt Correction vs Azimuth', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Absolute energy difference
        ax3 = axes[1, 0]
        ax3.scatter(valid_azimuths, benefit_kwh, alpha=0.7, color='orange', s=50)
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Array Azimuth (degrees)', fontsize=11)
        ax3.set_ylabel('Energy Difference (kWh)', fontsize=11)
        ax3.set_title('Absolute Terrain Benefit vs Azimuth', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Terrain explanation panel
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        max_benefit_idx = np.argmax(benefit_pct)
        best_azimuth = valid_azimuths[max_benefit_idx]
        max_benefit = benefit_pct[max_benefit_idx]
        
        explanation_text = f"""TERRAIN CORRECTION ANALYSIS

Location: {lat:.6f}N, {lon:.6f}W

GROUND SLOPE CHARACTERISTICS:
   Slope magnitude:     {terrain_data['slope']:.1f}
   Slope direction:     {terrain_data['aspect']:.1f} ({self._get_cardinal_direction(terrain_data['aspect'])})
   
CORRECTION STATISTICS:
   Max benefit:         {max_benefit:+.2f}% at {best_azimuth:.0f}
   Min benefit:         {benefit_pct.min():+.2f}%
   Benefit range:       {benefit_pct.max() - benefit_pct.min():.2f}%
   
   Max energy gain:     {benefit_kwh.max():+.1f} kWh/year
   Min energy gain:     {benefit_kwh.min():+.1f} kWh/year
   
   Max tilt correction: +/-{np.abs(tilt_corrections).max():.1f}
   Avg tilt correction: {np.mean(np.abs(tilt_corrections)):.1f}

PHYSICAL INTERPRETATION:
   * {self._get_cardinal_direction(terrain_data['aspect'])}-facing slope affects all orientations
   * Terrain corrections are geometric projections
   * Benefits depend on solar geometry interaction
   * Small corrections -> modest but measurable gains"""
        
        ax4.text(0.02, 0.98, explanation_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Terrain Correction Details\nGround Slope: {terrain_data["slope"]:.1f}° toward {self._get_cardinal_direction(terrain_data["aspect"])}', 
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save terrain details plot
        terrain_filename = f'terrain_details_{lat:.3f}_{lon:.3f}.png'
        plt.savefig(terrain_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Terrain details saved as '{terrain_filename}'")
    
    def compare_multiple_locations(self, locations, config=None):
        """
        Compare optimal configurations across multiple locations.
        
        Parameters:
        -----------
        locations : list of tuples
            [(lat1, lon1, name1), (lat2, lon2, name2), ...]
        config : dict, optional
            Optimization configuration (tilt_range, azimuth_range, step_size)
            
        Returns:
        --------
        dict : Comparison results
        """
        
        print(f"\n🗺️ Multi-Location Terrain Analysis")
        print(f"=" * 40)
        
        if config is None:
            config = {'tilt_range': (20, 40), 'azimuth_range': (160, 200), 'step_size': 5}
        
        comparison_results = []
        
        for lat, lon, name in locations:
            try:
                print(f"\n📍 Analyzing {name}...")
                result = self.optimize_location(lat, lon, detailed_output=False, **config)
                result['location_name'] = name
                comparison_results.append(result)
                
                print(f"   Optimal: {result['optimal_tilt']:.1f}°/{result['optimal_azimuth']:.0f}°")
                print(f"   Energy: {result['max_energy']:.0f} kWh/year")
                print(f"   Terrain benefit: {result['terrain_benefit_pct']:+.2f}%")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        # Create azimuth curves for first location as example
        if comparison_results:
            print(f"\n📈 Creating azimuth analysis for first location...")
            first_location = locations[0]
            self.plot_azimuth_energy_curve(first_location[0], first_location[1], 
                                         tilt=config.get('tilt_range', (30, 30))[0])
            
        return comparison_results
    
    def create_terrain_suitability_map(self, grid_resolution=50):
        """
        Create a map showing PV suitability across the terrain area.
        """
        
        print(f"\n🗺️ Creating Terrain Suitability Map")
        print(f"=" * 40)
        
        # Create analysis grid
        x_min, x_max = self.analyzer.bounds['x_min'], self.analyzer.bounds['x_max']
        y_min, y_max = self.analyzer.bounds['y_min'], self.analyzer.bounds['y_max']
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        
        suitability_grid = np.full((grid_resolution, grid_resolution), np.nan)
        
        print(f"   Grid resolution: {grid_resolution} × {grid_resolution}")
        print(f"   Analyzing {grid_resolution**2:,} points...")
        
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                try:
                    # Convert to lat/lon for optimization
                    lat, lon = self.analyzer.convert_utm_to_latlon(x, y)
                    
                    # Quick optimization (coarse grid)
                    result = self.optimize_location(
                        lat, lon, 
                        tilt_range=(25, 35), 
                        azimuth_range=(170, 190),
                        step_size=10, 
                        detailed_output=False
                    )
                    
                    # Use energy density as suitability metric
                    suitability_grid[j, i] = result['max_energy']
                    
                except:
                    # Outside coverage or failed optimization
                    suitability_grid[j, i] = np.nan
        
        # Create visualization
        self._plot_suitability_map(x_grid, y_grid, suitability_grid)
        
        return suitability_grid
    
    def _get_terrain_characteristics(self, x_utm, y_utm):
        """Get terrain data for a specific UTM location."""
        
        distances = np.sqrt((self.analyzer.x_coords - x_utm)**2 + (self.analyzer.y_coords - y_utm)**2)
        nearest_idx = np.argmin(distances)
        
        return {
            'x': self.analyzer.x_coords[nearest_idx],
            'y': self.analyzer.y_coords[nearest_idx],
            'elevation': self.analyzer.elevations[nearest_idx],
            'slope': self.analyzer.grid_points[nearest_idx]['slope'],
            'aspect': self.analyzer.grid_points[nearest_idx]['aspect'],
            'distance_to_query': distances[nearest_idx]
        }
    
    def _get_weather_data(self, lat, lon):
        """Get weather data for energy simulation (matching previous analysis)."""
        
        # Use cached weather if available for this general area
        cache_key = f"{lat:.2f}_{lon:.2f}"
        
        if cache_key not in self._weather_cache:
            site = location.Location(latitude=lat, longitude=lon, tz=self.timezone)
            times = pd.date_range('2023-01-01', '2023-12-31', freq='h', tz=site.tz)
            
            clearsky = site.get_clearsky(times)
            weather = clearsky.copy()
            weather['temp_air'] = 20  # More realistic average
            weather['wind_speed'] = 3  # More realistic average
            
            self._weather_cache[cache_key] = weather
        
        return self._weather_cache[cache_key]
    
    def _simulate_energy(self, site, weather, effective_tilt, effective_azimuth):
        """Run pvlib energy simulation for given configuration."""
        
        system = pvsystem.PVSystem(
            surface_tilt=effective_tilt,
            surface_azimuth=effective_azimuth,
            module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
            inverter_parameters={'pdc0': 300, 'eta_inv_nom': 0.96},
            racking_model='open_rack',
            module_type='glass_glass'
        )
        
        mc = modelchain.ModelChain(
            system, site,
            aoi_model='no_loss',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
        )
        
        mc.run_model(weather)
        return mc.results.ac.sum() / 1000  # kWh
    
    def _analyze_optimization_results(self, results, best_config, terrain_data, detailed_output):
        """Analyze and summarize optimization results."""
        
        # Find baseline (south-facing, latitude-optimized)
        baseline_tilt = 30  # Rough latitude optimization for 39°N
        baseline_azimuth = 180  # South
        
        baseline_result = None
        for r in results:
            if (abs(r['nominal_tilt'] - baseline_tilt) < 2.5 and 
                abs(r['nominal_azimuth'] - baseline_azimuth) < 2.5):
                baseline_result = r
                break
        
        # Calculate improvements
        improvement_kwh = 0
        improvement_pct = 0
        
        if baseline_result:
            improvement_kwh = best_config['annual_energy_kwh'] - baseline_result['annual_energy_kwh']
            improvement_pct = (improvement_kwh / baseline_result['annual_energy_kwh']) * 100
        
        # Terrain benefit calculation
        # Compare with flat-ground simulation
        try:
            flat_energy = self._simulate_energy(
                location.Location(latitude=0, longitude=0, tz=self.timezone),
                self._get_weather_data(39.8, -79.1),  # Approximate location
                best_config['nominal_tilt'], 
                best_config['nominal_azimuth']
            )
            terrain_benefit_kwh = best_config['annual_energy_kwh'] - flat_energy
            terrain_benefit_pct = (terrain_benefit_kwh / flat_energy) * 100
        except:
            terrain_benefit_kwh = 0
            terrain_benefit_pct = 0
        
        if detailed_output:
            print(f"\n📊 Optimization Results:")
            print(f"   🏆 OPTIMAL CONFIGURATION:")
            print(f"      Nominal: {best_config['nominal_tilt']:.1f}°/{best_config['nominal_azimuth']:.0f}°")
            print(f"      Effective: {best_config['effective_tilt']:.1f}°/{best_config['effective_azimuth']:.1f}°")
            print(f"      Annual energy: {best_config['annual_energy_kwh']:.0f} kWh")
            
            print(f"\n   📈 Performance Comparison:")
            if baseline_result:
                print(f"      vs Traditional (30°/180°): {improvement_pct:+.2f}% ({improvement_kwh:+.1f} kWh)")
            print(f"      Terrain benefit: {terrain_benefit_pct:+.2f}% ({terrain_benefit_kwh:+.1f} kWh)")
            
            print(f"\n   🏔️ Terrain Corrections:")
            print(f"      Tilt correction: {best_config['tilt_correction']:+.1f}°")
            print(f"      Azimuth correction: {best_config['azimuth_correction']:+.1f}°")
        
        return {
            'optimal_tilt': best_config['nominal_tilt'],
            'optimal_azimuth': best_config['nominal_azimuth'],
            'effective_tilt': best_config['effective_tilt'],
            'effective_azimuth': best_config['effective_azimuth'],
            'max_energy': best_config['annual_energy_kwh'],
            'tilt_correction': best_config['tilt_correction'],
            'azimuth_correction': best_config['azimuth_correction'],
            'improvement_vs_traditional_pct': improvement_pct,
            'terrain_benefit_pct': terrain_benefit_pct,
            'terrain_data': terrain_data,
            'all_results': results
        }
    
    def _get_cardinal_direction(self, azimuth):
        """Convert azimuth to cardinal direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = round(azimuth / 22.5) % 16
        return directions[idx]
    
    def _plot_suitability_map(self, x_grid, y_grid, suitability_grid):
        """Create terrain suitability visualization."""
        
        plt.figure(figsize=(12, 8))
        
        # Create contour plot
        plt.contourf(x_grid, y_grid, suitability_grid, levels=20, cmap='RdYlGn')
        plt.colorbar(label='Annual Energy (kWh)')
        
        plt.xlabel('UTM Easting (m)')
        plt.ylabel('UTM Northing (m)')
        plt.title('PV Energy Potential Map\n(Terrain-Optimized Arrays)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('terrain_suitability_map.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Suitability map saved as 'terrain_suitability_map.png'")


def example_usage():
    """Demonstrate the terrain-aware optimization tool."""
    
    print("🚀 Terrain-Aware PV Optimization Tool Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = TerrainAwarePVOptimizer('data/terrain_elevation_points.geojson')
    
    # Example 1: Create azimuth vs energy curve with adaptive spacing
    print("\n📍 EXAMPLE 1: Azimuth vs Energy Curve Analysis (Adaptive Spacing)")
    curve_data = optimizer.plot_azimuth_energy_curve(
        39.796678, -79.092463,
        tilt=30,  # Fixed tilt
        azimuth_range=(160, 200),  # Focused range around south
        step_size=5,  # Base step size (for edges)
        show_terrain_benefit=True,
        adaptive_spacing=True  # 1° resolution near 180°
    )
    
    # Example 2: Show how to analyze a different location in the area
    print("\n📍 EXAMPLE 2: Different Location Analysis")
    try:
        # Analyze a slightly different location to show terrain variation
        curve_data_2 = optimizer.plot_azimuth_energy_curve(
            39.800000, -79.090000,
            tilt=30,
            azimuth_range=(160, 200),  # Focused range
            step_size=10,  # Coarser grid for speed
            show_terrain_benefit=True
        )
        
        print(f"\n🔍 Location Comparison:")
        print(f"   Location 1 optimal: {curve_data['optimal_corrected_azimuth']:.0f}° ({curve_data['max_energy_corrected']:.0f} kWh)")
        print(f"   Location 2 optimal: {curve_data_2['optimal_corrected_azimuth']:.0f}° ({curve_data_2['max_energy_corrected']:.0f} kWh)")
        
    except Exception as e:
        print(f"Second location analysis failed: {e}")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"The tool can now optimize any location within the terrain coverage area!")

if __name__ == "__main__":
    example_usage()