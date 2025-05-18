# JWST Atmospheric Composition Analyzer

Demo for Astronomical Society of Geelong.
A Python application to analyze astronomical data from the JWST API, focusing on planetary atmospheric composition.

## Features

- Fetch observations from the JWST API
- Analyze spectral data to detect atmospheric molecules
- Visualize spectral signatures with interactive plots
- Calculate confidence levels for detected chemical compounds
- Generate composition reports for exoplanet atmospheres

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from jwst_analyzer import JWSTDataAnalyzer

# Initialize with your API key
analyzer = JWSTDataAnalyzer(api_key="your_api_key_here")

# Fetch recent exoplanet observations
observations = analyzer.fetch_observations(category="exoplanets", limit=5)

# Get details for a specific observation
details = analyzer.fetch_observation_details(observation_id="example_id")

# Search for data related to a specific planet or molecule
search_results = analyzer.search_atmospheric_data(planet_name="WASP-39b", molecule="H2O")

# Analyze and visualize the data
spectral_data = analyzer.parse_spectral_data(details)
analyzer.plot_spectrum(spectral_data, title="WASP-39b Atmospheric Spectrum")
composition = analyzer.analyze_atmospheric_composition(spectral_data)
analyzer.plot_composition_bar(composition)
analyzer.interactive_plot(spectral_data)
```

### Demo Mode

You can run the included demo with mock data:

```
python jwst_analyzer.py
```

## API Key

To use real data from the JWST API, you need to:

1. Sign up at [https://jwstapi.com/](https://jwstapi.com/)
2. Obtain your API key
3. Pass it to the JWSTDataAnalyzer constructor

## Molecules Detected

The analyzer can identify spectral signatures for the following molecules:

- Water (H2O)
- Carbon Dioxide (CO2)
- Methane (CH4)
- Carbon Monoxide (CO)
- Ammonia (NH3)
- Sodium (Na)
- Potassium (K)

## Note

This is a basic implementation meant for educational purposes. Real atmospheric analysis requires complex radiative transfer models and advanced spectral processing techniques.

## License

MIT
