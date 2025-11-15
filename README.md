# ☀️ Solar Slope Scope

**Interactive terrain-aware solar panel azimuth optimization tool**

> ⚠️ **Disclaimer**: This whole thing was vibe coded. Don't judge me. While functional and based on solid solar engineering principles, use results as guidance rather than professional engineering analysis. Always consult qualified solar engineers for actual installations.

## 🎯 What It Does

This tool analyzes how ground slope affects optimal solar panel orientation. Instead of assuming flat ground, it uses real USGS terrain data to calculate how local topography should influence solar panel azimuth angles for maximum energy production.

**Key Features:**
- **Interactive satellite map** for site selection
- **Real terrain analysis** using USGS elevation data
- **Slope visualization** with directional gradients
- **Energy optimization** comparing flat vs. terrain-aware orientations
- **Elevation contour maps** overlaid on satellite imagery
- **Terrain impact assessment** with cardinal direction analysis

## 🌍 Live Demo

**[Deployed Streamlit Community](https://solar-slope-scope-fokgwoteq5dyzlrubrqvxp.streamlit.app/)**

## 📍 Coverage Area

Currently configured for **Meyersdale, Pennsylvania** region:
- **Coordinates**: 39.796678°N, 79.092463°W  
- **Coverage**: ~800m × 1000m area
- **Resolution**: 5m × 5m grid (32,436 elevation points)
- **Elevation range**: 676-723m (47m total relief)
- **Terrain type**: Rolling hills with moderate slopes

## 🏗️ How It Works

### Solar Analysis
- Uses **pvlib-python** for accurate solar position and energy modeling
- Calculates optimal azimuth angles for different ground slopes
- Compares energy output between flat-ground and terrain-aware orientations
- Provides adaptive azimuth spacing for detailed analysis around optimal angles

### Terrain Processing
- **USGS elevation data**: High-resolution 1m pixel terrain data
- **Coordinate conversion**: UTM Zone 17N ↔ lat/lon transformations
- **Slope calculation**: Trigonometric analysis of local terrain gradients
- **Interpolation**: Smooth terrain surface generation for analysis

### Interactive Features
- **Click-to-analyze**: Select any point on the satellite map
- **Real-time feedback**: Immediate terrain analysis and slope visualization
- **Energy curves**: Interactive plots showing azimuth vs. energy output
- **Terrain explainer**: Built-in guide for interpreting slope analysis

## 🛠️ Technical Stack

- **pvlib-python**: Solar energy modeling and weather data
- **Streamlit**: Web application framework
- **Folium**: Interactive satellite maps with terrain overlays
- **Plotly**: Interactive energy analysis charts
- **GeoPandas**: Geospatial data processing
- **NumPy/SciPy**: Numerical analysis and interpolation
- **GDAL/OGR**: Geospatial data reading and coordinate transformations

## 📐 Array Specifications

Optimized for typical ground-mounted solar installations:
- **Array size**: 55' × 15' (16.8m × 4.6m)
- **Orientation**: 55' dimension runs E-W when facing south
- **Tilt mechanism**: Panels tilt along the long (55') axis
- **Default tilt**: 30° (adjustable in interface)

## 🚀 Local Setup

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/solar-slope-scope.git
cd solar-slope-scope

# Create conda environment
conda create -n pv python=3.11
conda activate pv

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_pv_app.py
```

### Data Requirements
The terrain elevation data (`data/terrain_elevation_points.geojson`) is included in the repository. For other regions, you would need:
- USGS elevation data in GeoJSON format
- 5m × 5m grid spacing recommended
- UTM coordinate system for accurate distance calculations

## 🔬 Technical Details

### Coordinate Systems
- **Display**: WGS84 Geographic (lat/lon)
- **Analysis**: UTM Zone 17N for accurate spatial calculations
- **Conversion**: Automatic transformation between systems

### Slope Analysis
- **Local gradients**: Calculated in cardinal directions (N/E/S/W)
- **USGS integration**: Uses official slope and aspect values
- **Directional analysis**: Shows uphill/downhill trends around each point
- **Visual feedback**: Arrow indicators and slope vector diagrams

### Energy Modeling
- **Weather data**: Uses pvlib's built-in databases
- **System parameters**: Configurable tilt and azimuth ranges
- **Optimization**: Adaptive grid spacing for efficient analysis
- **Comparison**: Side-by-side flat vs. terrain-corrected results

## 📊 Example Results

Typical analysis shows:
- **Slope impact**: 2-5° azimuth adjustments for moderate slopes (5-10°)
- **Energy difference**: 1-3% improvement with terrain-aware optimization
- **Regional patterns**: South-facing slopes benefit most from azimuth adjustments

## ⚠️ Limitations

- **Regional coverage**: Currently limited to Meyersdale, PA area
- **Resolution**: 5m grid may miss micro-topography effects
- **Weather data**: Uses typical meteorological year, not real-time conditions
- **Shading**: Does not account for vegetation, buildings, or other obstructions
- **Engineering**: Results are for analysis only, not professional design specifications

## 🔮 Future Enhancements

- **Expanded coverage**: Additional regions with USGS data integration
- **Shading analysis**: Integration with building and vegetation databases
- **Seasonal optimization**: Month-by-month azimuth recommendations
- **Batch processing**: Multiple site analysis and comparison tools
- **Export features**: PDF reports and data download capabilities

## 🤝 Contributing

This project welcomes contributions! Areas of interest:
- Additional geographic regions
- Enhanced terrain visualization
- Performance optimizations
- Mobile-responsive design
- API development for batch processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

