import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from terrain_aware_optimizer import TerrainAwarePVOptimizer
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Configure Streamlit page
st.set_page_config(
    page_title="PV Terrain Optimizer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

@st.cache_resource
def load_optimizer():
    """Load the terrain optimizer (cached for performance)"""
    try:
        optimizer = TerrainAwarePVOptimizer('data/terrain_elevation_points.geojson')
        return optimizer, None
    except Exception as e:
        return None, str(e)

def create_coverage_map(optimizer):
    """Create interactive map showing coverage area with click selection"""
    
    # Get coverage bounds
    bounds = optimizer.analyzer.bounds
    
    # Convert UTM bounds to lat/lon for map display
    lat_min, lon_min = optimizer.analyzer.convert_utm_to_latlon(bounds['x_min'], bounds['y_min'])
    lat_max, lon_max = optimizer.analyzer.convert_utm_to_latlon(bounds['x_max'], bounds['y_max'])
    
    # Center of coverage area
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    # Create folium map with satellite tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None  # We'll add custom tiles
    )
    
    # Add satellite tile layer as default
    satellite_layer = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri WorldImagery',
        name='Satellite',
        overlay=False,
        control=True
    )
    satellite_layer.add_to(m)
    
    # Add OpenStreetMap as secondary option
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add coverage area rectangle (more transparent, non-clickable)
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color='cyan',
        weight=2,
        fill=True,
        fillColor='lightblue',
        fillOpacity=0.15,
        popup=None,  # Remove popup to prevent interference
        tooltip='Terrain Analysis Coverage Area'
    ).add_to(m)
    
    # Add center marker for reference
    folium.Marker(
        [center_lat, center_lon],
        popup='Coverage Center',
        tooltip='Coverage area center - click anywhere in blue area to analyze',
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add terrain visualization (try contours, fallback to heatmap)
    contour_success = False
    try:
        m = add_contour_lines_to_map(m, optimizer)
        contour_success = True
    except Exception as e:
        print(f"Contour lines failed: {e}")
    
    # If contours failed, add elevation heatmap instead
    if not contour_success:
        try:
            m = add_elevation_heatmap_to_map(m, optimizer)
            print("Added elevation heatmap instead of contours")
        except Exception as e:
            print(f"Both terrain visualizations failed: {e}")
    
    return m

def create_terrain_summary_widget(optimizer, lat, lon):
    """Create a fast terrain summary display for a specific location"""
    
    # Convert to UTM for analysis
    x_utm, y_utm = optimizer.analyzer.convert_latlon_to_utm(lat, lon)
    
    # Get terrain data
    terrain_data = optimizer._get_terrain_characteristics(x_utm, y_utm)
    
    # Get elevation at point
    from scipy.interpolate import griddata
    points = np.array([[pt['x'], pt['y']] for pt in optimizer.analyzer.grid_points])
    values = np.array([pt['elevation'] for pt in optimizer.analyzer.grid_points])
    elevation = griddata(points, values, (x_utm, y_utm), method='linear')
    
    # Calculate gradient in cardinal directions (simplified)
    radius = 25  # 25m radius for local analysis
    directions = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
    gradients = {}
    
    for name, bearing in directions.items():
        rad = np.radians(bearing)
        sample_x = x_utm + radius * np.sin(rad)
        sample_y = y_utm + radius * np.cos(rad)
        
        try:
            sample_elev = griddata(points, values, (sample_x, sample_y), method='linear')
            if not np.isnan(sample_elev):
                elevation_diff = sample_elev - elevation
                slope = np.degrees(np.arctan(elevation_diff / radius))
                gradients[name] = slope
            else:
                gradients[name] = 0
        except:
            gradients[name] = 0
    
    # Create a simple arrow visualization showing slope direction
    fig = go.Figure()
    
    # Add the main slope vector
    slope_mag = terrain_data['slope']
    slope_dir = terrain_data['aspect']
    
    # Convert aspect to compass bearing for display
    arrow_x = slope_mag * np.sin(np.radians(slope_dir))
    arrow_y = slope_mag * np.cos(np.radians(slope_dir))
    
    fig.add_trace(go.Scatter(
        x=[0, arrow_x],
        y=[0, arrow_y],
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(color='red', size=[8, 12], symbol=['circle', 'triangle-up']),
        name=f'Slope Vector ({slope_mag:.1f}°)',
        hovertemplate=f'Direction: {slope_dir:.0f}° ({optimizer._get_cardinal_direction(slope_dir)})<br>Magnitude: {slope_mag:.1f}°<extra></extra>'
    ))
    
    # Add cardinal direction reference
    for name, bearing in directions.items():
        ref_x = 8 * np.sin(np.radians(bearing))
        ref_y = 8 * np.cos(np.radians(bearing))
        
        fig.add_annotation(
            x=ref_x, y=ref_y,
            text=name,
            showarrow=False,
            font=dict(size=12, color='gray')
        )
    
    fig.update_layout(
        title=f"Terrain Slope: {slope_mag:.1f}° toward {optimizer._get_cardinal_direction(slope_dir)}",
        xaxis=dict(range=[-10, 10], showgrid=True, zeroline=True, title=""),
        yaxis=dict(range=[-10, 10], showgrid=True, zeroline=True, title=""),
        height=250,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig, terrain_data, elevation, gradients

def add_elevation_heatmap_to_map(m, optimizer):
    """Add elevation heatmap overlay to the folium map as an alternative to contours"""
    
    try:
        # Get terrain data bounds
        bounds = optimizer.analyzer.bounds
        
        # Create a grid for heatmap (smaller for performance)
        resolution = 15
        x_grid = np.linspace(bounds['x_min'], bounds['x_max'], resolution)
        y_grid = np.linspace(bounds['y_min'], bounds['y_max'], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Use griddata for interpolation
        from scipy.interpolate import griddata
        
        # Prepare data points
        points = np.array([[pt['x'], pt['y']] for pt in optimizer.analyzer.grid_points])
        values = np.array([pt['elevation'] for pt in optimizer.analyzer.grid_points])
        
        # Interpolate elevations on the grid
        Z = griddata(points, values, (X, Y), method='linear')
        
        # Convert UTM coordinates back to lat/lon for folium
        heat_data = []
        for i in range(resolution):
            for j in range(resolution):
                if not np.isnan(Z[i, j]):
                    lat, lon = optimizer.analyzer.convert_utm_to_latlon(X[i, j], Y[i, j])
                    # Normalize elevation to 0-1 range for heatmap intensity
                    normalized_elev = (Z[i, j] - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                    heat_data.append([lat, lon, normalized_elev])
        
        # Add heatmap
        from folium.plugins import HeatMap
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.0: '#0066CC', 0.3: '#00AA44', 0.6: '#FFAA00', 1.0: '#CC0000'}
        ).add_to(m)
        
    except Exception as e:
        print(f"Could not add elevation heatmap: {e}")
    
    return m

def add_contour_lines_to_map(m, optimizer):
    """Add elevation contour lines to the folium map"""
    
    try:
        # Get terrain data bounds
        bounds = optimizer.analyzer.bounds
        
        # Create a coarser grid for performance (20x20)
        x_grid = np.linspace(bounds['x_min'], bounds['x_max'], 20)
        y_grid = np.linspace(bounds['y_min'], bounds['y_max'], 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Use griddata for interpolation
        from scipy.interpolate import griddata
        
        # Prepare data points
        points = np.array([[pt['x'], pt['y']] for pt in optimizer.analyzer.grid_points])
        values = np.array([pt['elevation'] for pt in optimizer.analyzer.grid_points])
        
        # Interpolate elevations on the grid
        Z = griddata(points, values, (X, Y), method='linear')
        
        # Convert UTM coordinates back to lat/lon for folium
        lat_grid = np.zeros_like(X)
        lon_grid = np.zeros_like(X)
        
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                lat_grid[i, j], lon_grid[i, j] = optimizer.analyzer.convert_utm_to_latlon(X[i, j], Y[i, j])
        
        # Generate more contour levels for better detail
        elevation_range = np.nanmax(Z) - np.nanmin(Z)
        # Use more levels based on elevation range - aim for ~5-10m spacing
        num_levels = max(8, min(15, int(elevation_range / 5)))  # 8-15 levels, ~5m spacing
        contour_levels = np.linspace(np.nanmin(Z), np.nanmax(Z), num_levels)
        
        # Create contours using contour (not contourf)
        fig_temp, ax_temp = plt.subplots(figsize=(1, 1))
        cs = ax_temp.contour(lon_grid, lat_grid, Z, levels=contour_levels)
        
        # Extract contour paths and add to folium map
        # Generate colors from dark brown to light tan
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        for i, level_collection in enumerate(cs.allsegs):
            # Use terrain colormap for natural elevation colors
            color_intensity = i / (len(cs.allsegs) - 1) if len(cs.allsegs) > 1 else 0.5
            color_rgb = cm.terrain(0.3 + 0.4 * color_intensity)  # Use middle portion of terrain colormap
            color = mcolors.to_hex(color_rgb)
            
            for contour_path in level_collection:
                if len(contour_path) > 2:  # Skip very short segments
                    # contour_path is array of [lon, lat] points
                    locations = [(lat, lon) for lon, lat in contour_path]
                    
                    folium.PolyLine(
                        locations=locations,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=f'Elevation: {contour_levels[i]:.1f}m'
                    ).add_to(m)
        
        plt.close(fig_temp)  # Clean up temporary figure
        
    except Exception as e:
        print(f"Could not add contour lines: {e}")
    
    return m

def create_plotly_energy_curve(curve_data, lat, lon):
    """Create interactive Plotly energy vs azimuth curve"""
    
    # Create single plot for energy curves only
    fig = go.Figure()
    
    # Main energy curves
    fig.add_trace(
        go.Scatter(
            x=curve_data['azimuths'],
            y=curve_data['energies_nominal'],
            mode='lines',
            name='Flat Ground (Nominal)',
            line=dict(color='lightblue', width=2),
            hovertemplate='Azimuth: %{x}°<br>Energy: %{y:.0f} kWh/year<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=curve_data['azimuths'],
            y=curve_data['energies_corrected'],
            mode='lines',
            name='Terrain-Corrected',
            line=dict(color='darkblue', width=3),
            hovertemplate='Azimuth: %{x}°<br>Energy: %{y:.0f} kWh/year<extra></extra>'
        )
    )
    
    # Mark optimal points
    fig.add_trace(
        go.Scatter(
            x=[curve_data['optimal_nominal_azimuth']],
            y=[curve_data['max_energy_nominal']],
            mode='markers',
            name=f"Optimal (Flat): {curve_data['optimal_nominal_azimuth']:.0f}°",
            marker=dict(color='lightblue', size=10, symbol='star'),
            hovertemplate=f"Optimal Flat: {curve_data['optimal_nominal_azimuth']:.0f}°<br>Energy: {curve_data['max_energy_nominal']:.0f} kWh/year<extra></extra>"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[curve_data['optimal_corrected_azimuth']],
            y=[curve_data['max_energy_corrected']],
            mode='markers',
            name=f"Optimal (Terrain): {curve_data['optimal_corrected_azimuth']:.0f}°",
            marker=dict(color='red', size=12, symbol='star'),
            hovertemplate=f"Optimal Terrain: {curve_data['optimal_corrected_azimuth']:.0f}°<br>Energy: {curve_data['max_energy_corrected']:.0f} kWh/year<extra></extra>"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"PV Energy Analysis: {lat:.4f}°N, {lon:.4f}°W",
        height=500,
        xaxis_title="Azimuth (degrees)",
        yaxis_title="Energy (kWh/year)",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return fig

def run_azimuth_analysis(optimizer, lat, lon, tilt, azimuth_range, step_size, adaptive_spacing):
    """Run azimuth analysis without matplotlib file generation"""
    
    print(f"\n📈 Running Azimuth Analysis for Web App")
    print(f"📍 Location: {lat:.6f}°N, {lon:.6f}°W")
    print(f"📐 Fixed tilt: {tilt}°")
    print(f"🫴 Azimuth range: {azimuth_range[0]}° to {azimuth_range[1]}°")
    print(f"📐 Adaptive spacing: {adaptive_spacing}")
    
    # Get terrain characteristics
    x_utm, y_utm = optimizer.analyzer.convert_latlon_to_utm(lat, lon)
    
    if not (optimizer.analyzer.bounds['x_min'] <= x_utm <= optimizer.analyzer.bounds['x_max'] and
            optimizer.analyzer.bounds['y_min'] <= y_utm <= optimizer.analyzer.bounds['y_max']):
        raise ValueError(f"Location outside terrain coverage area")
    
    terrain_data = optimizer._get_terrain_characteristics(x_utm, y_utm)
    
    print(f"🏔️ Ground slope: {terrain_data['slope']:.1f}° toward {terrain_data['aspect']:.1f}° ({optimizer._get_cardinal_direction(terrain_data['aspect'])})")
    
    # Create azimuth array with adaptive spacing
    if adaptive_spacing:
        # Fine resolution (1°) around south (170-190°)
        # Coarse resolution at the edges
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
    weather = optimizer._get_weather_data(lat, lon)
    
    # Calculate energy for each azimuth
    energies_nominal = []
    energies_corrected = []
    tilt_corrections = []
    azimuth_corrections = []
    
    print(f"⚡ Calculating energy for {len(azimuths)} orientations...")
    
    from pvlib import location as pvlib_location
    site = pvlib_location.Location(latitude=lat, longitude=lon, tz=optimizer.timezone)
    
    for azimuth in azimuths:
        try:
            # Calculate terrain-corrected geometry
            from ground_slope_analysis import calculate_effective_array_geometry
            effective_geom = calculate_effective_array_geometry(
                terrain_data['slope'], terrain_data['aspect'], tilt, azimuth
            )
            
            # Nominal energy (flat ground)
            energy_nominal = optimizer._simulate_energy(
                site, weather, tilt, azimuth
            )
            
            # Terrain-corrected energy
            energy_corrected = optimizer._simulate_energy(
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
    
    # Calculate ranges
    energy_range_nominal = np.nanmax(energies_nominal) - np.nanmin(energies_nominal)
    energy_range_corrected = np.nanmax(energies_corrected) - np.nanmin(energies_corrected)
    
    print(f"✅ Analysis complete!")
    
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
        'energy_range_corrected': energy_range_corrected
    }

def main():
    """Main Streamlit app"""
    
    # App header
    st.title("☀️ PV Terrain-Aware Azimuth Optimizer")
    st.markdown("""
    **Interactive tool for optimizing solar panel orientation based on local terrain**
    - Click anywhere on the satellite map to select a location
    - Analyze how ground slope affects optimal solar panel azimuth
    - Compare energy output for different orientations
    """)
    
    # Load optimizer
    optimizer, error = load_optimizer()
    if optimizer is None:
        st.error(f"❌ Failed to load terrain data: {error}")
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Analysis Parameters")
        
        tilt_angle = st.slider(
            "Array Tilt Angle (degrees)", 
            min_value=15, max_value=45, value=30, step=1,
            help="Fixed tilt angle for the solar array"
        )
        
        azimuth_min = st.number_input(
            "Min Azimuth (degrees)", 
            min_value=0, max_value=360, value=160, step=5
        )
        
        azimuth_max = st.number_input(
            "Max Azimuth (degrees)", 
            min_value=0, max_value=360, value=200, step=5
        )
        
        step_size = st.selectbox(
            "Step Size (degrees)",
            options=[1, 2, 5, 10],
            index=2,
            help="Coarse spacing at edges (fine spacing of 1° around 180° is automatic)"
        )
        
        adaptive_spacing = st.checkbox(
            "Adaptive Spacing",
            value=True,
            help="Use fine resolution (1°) around south-facing (170-190°), coarser at edges"
        )
    
    # Map section - full width
    st.subheader("🗺️ Select Analysis Location")
    
    # Create two columns for map and controls
    map_col, controls_col = st.columns([2, 1])
    
    with map_col:
        # Create and display map
        m = create_coverage_map(optimizer)
        
        # Add clicked marker if coordinates exist in session state
        if hasattr(st.session_state, 'clicked_lat') and hasattr(st.session_state, 'clicked_lon'):
            folium.Marker(
                [st.session_state.clicked_lat, st.session_state.clicked_lon],
                popup=f'Selected: {st.session_state.clicked_lat:.6f}°N, {st.session_state.clicked_lon:.6f}°W',
                tooltip='Selected Analysis Point',
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)
        
        map_data = st_folium(m, width=800, height=600, returned_objects=["last_clicked"])
    
    with controls_col:
        # Handle clicked coordinates
        clicked_lat, clicked_lon = None, None
        if map_data['last_clicked'] is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            # Store in session state for persistent marker
            st.session_state.clicked_lat = clicked_lat
            st.session_state.clicked_lon = clicked_lon
            # Force rerun to show marker immediately
            st.rerun()
        
        # Use session state coordinates if available
        if hasattr(st.session_state, 'clicked_lat'):
            clicked_lat = st.session_state.clicked_lat
            clicked_lon = st.session_state.clicked_lon
        
        # Display selected coordinates and validation
        if clicked_lat and clicked_lon:
            st.success(f"📍 Clicked: {clicked_lat:.6f}°N, {clicked_lon:.6f}°W")
            
            # Validate location is within coverage
            try:
                x_utm, y_utm = optimizer.analyzer.convert_latlon_to_utm(clicked_lat, clicked_lon)
                bounds = optimizer.analyzer.bounds
                
                if (bounds['x_min'] <= x_utm <= bounds['x_max'] and
                    bounds['y_min'] <= y_utm <= bounds['y_max']):
                    
                    # Location is valid - show analysis button
                    if st.button("🚀 Run PV Analysis", type="primary"):
                        st.session_state.selected_lat = clicked_lat
                        st.session_state.selected_lon = clicked_lon
                        st.session_state.run_analysis = True
                        st.rerun()
                else:
                    st.error("❌ Location outside coverage area")
                    st.info("Please click within the light blue rectangle")
                    
            except Exception as e:
                st.error(f"❌ Invalid location: {e}")
        else:
            st.info("👆 Click anywhere on the map to select a location")
        
        # Manual coordinate input
        with st.expander("📝 Manual Coordinates"):
            manual_lat = st.number_input("Latitude", value=39.796678, format="%.6f")
            manual_lon = st.number_input("Longitude", value=-79.092463, format="%.6f")
            
            if st.button("Use Manual Coordinates"):
                st.session_state.selected_lat = manual_lat
                st.session_state.selected_lon = manual_lon
                st.session_state.run_analysis = True
                st.rerun()
        
        # Terrain analysis for selected point
        if clicked_lat and clicked_lon:
            st.markdown("---")
            st.subheader("🏔️ Local Terrain Analysis")
            
            # Add explainer
            with st.expander("📚 How to Interpret the Terrain Analysis"):
                st.markdown("""
                **Slope Vector Diagram:**
                - The red arrow shows the direction and steepness of the ground slope
                - Arrow points "downhill" (direction water would flow)
                - Longer arrow = steeper slope
                - N/E/S/W labels show cardinal directions for reference
                
                **Local Gradients:**
                - Shows how elevation changes in each cardinal direction
                - ⬆️ Positive = uphill in that direction
                - ⬇️ Negative = downhill in that direction  
                - ➡️ Near zero = relatively flat
                
                **Why This Matters for Solar:**
                Ground slope affects the optimal panel orientation. Panels on south-facing slopes 
                may benefit from different azimuth angles than those on flat ground or north-facing slopes.
                """)
            
            try:
                # Create terrain summary
                terrain_fig, terrain_info, elevation, gradients = create_terrain_summary_widget(optimizer, clicked_lat, clicked_lon)
                st.plotly_chart(terrain_fig, width='stretch')
                
                # Display terrain metrics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 USGS Data:**")
                    st.markdown(f"• Slope: {terrain_info['slope']:.1f}°")
                    st.markdown(f"• Aspect: {terrain_info['aspect']:.0f}° ({optimizer._get_cardinal_direction(terrain_info['aspect'])})")
                    st.markdown(f"• Elevation: {elevation:.1f}m")
                
                with col2:
                    st.markdown("**📐 Local Gradients:**")
                    for direction, gradient in gradients.items():
                        arrow = "⬆️" if gradient > 0 else "⬇️" if gradient < 0 else "➡️"
                        st.markdown(f"• {direction}: {gradient:+.1f}° {arrow}")
                
            except Exception as e:
                st.error(f"Could not analyze terrain: {e}")
    
    # Results section - split layout
    st.divider()
    
    # Check if we should run analysis
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        selected_lat = st.session_state.selected_lat
        selected_lon = st.session_state.selected_lon
        
        try:
            with st.spinner('Running azimuth optimization...'):
                # Run analysis using new function
                curve_data = run_azimuth_analysis(
                    optimizer,
                    selected_lat, selected_lon,
                    tilt=tilt_angle,
                    azimuth_range=(azimuth_min, azimuth_max),
                    step_size=step_size,
                    adaptive_spacing=adaptive_spacing
                )
                
                # Store results
                st.session_state.last_analysis = {
                    'curve_data': curve_data,
                    'lat': selected_lat,
                    'lon': selected_lon
                }
                
                # Clear the run flag
                st.session_state.run_analysis = False
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    
    # Display results if available
    if hasattr(st.session_state, 'last_analysis') and st.session_state.last_analysis:
        
        # Results layout: 1/3 metrics, 2/3 plot
        metrics_col, plot_col = st.columns([1, 2])
        
        with metrics_col:
            st.subheader("📊 Analysis Results")
            
            analysis = st.session_state.last_analysis
            curve_data = analysis['curve_data']
            lat = analysis['lat']
            lon = analysis['lon']
            
            # Calculate terrain benefit
            terrain_benefit = ((curve_data['max_energy_corrected'] - curve_data['max_energy_nominal']) / curve_data['max_energy_nominal']) * 100
            
            # Display key metrics
            st.metric(
                "Optimal Azimuth", 
                f"{curve_data['optimal_corrected_azimuth']:.0f}°",
                delta=f"{curve_data['optimal_corrected_azimuth'] - 180:+.0f}° vs South"
            )
            
            st.metric(
                "Max Energy",
                f"{curve_data['max_energy_corrected']:.0f} kWh/year"
            )
            
            st.metric(
                "Terrain Benefit",
                f"{terrain_benefit:+.2f}%"
            )
            
            st.metric(
                "Ground Slope",
                f"{curve_data['terrain_data']['slope']:.1f}°",
                delta=f"toward {curve_data['terrain_data']['aspect']:.0f}°"
            )
            
            # Location info
            st.markdown("---")
            st.markdown(f"**Location:** {lat:.4f}°N, {lon:.4f}°W")
            st.markdown(f"**Tilt Angle:** {tilt_angle}°")
            st.markdown(f"**Analysis Range:** {azimuth_min}°-{azimuth_max}°")
        
        with plot_col:
            st.subheader("📈 Energy vs Azimuth")
            
            # Create and display energy plot
            energy_fig = create_plotly_energy_curve(curve_data, lat, lon)
            st.plotly_chart(energy_fig, width='stretch')
    else:
        st.info("👆 Select a location on the map and run analysis to see results")

if __name__ == "__main__":
    main()