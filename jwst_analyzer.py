import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class JWSTDataAnalyzer:
    """
    A class to analyze astronomical data from JWST API
    focusing on planetary atmospheric composition
    """
    
    def __init__(self, api_key=None):
        """Initialize with optional API key"""
        self.base_url = "https://api.jwstapi.com"
        self.api_key = api_key
        self.headers = {}
        
        if self.api_key:
            self.headers = {
                "X-API-KEY": self.api_key
            }
    
    def fetch_observations(self, category="exoplanets", limit=10):
        """Fetch recent observations related to exoplanets"""
        endpoint = f"/api/v1/observations/{category}"
        params = {
            "limit": limit
        }
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}", 
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_observation_details(self, observation_id):
        """Fetch details for a specific observation"""
        endpoint = f"/api/v1/observation/{observation_id}"
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}", 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching observation details: {e}")
            return None
    
    def search_atmospheric_data(self, planet_name=None, molecule=None, limit=10):
        """Search for atmospheric data by planet name or molecule"""
        endpoint = "/api/v1/search"
        params = {
            "limit": limit
        }
        
        if planet_name:
            params["query"] = planet_name
        
        if molecule:
            params["filters"] = f"molecule:{molecule}"
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}", 
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching data: {e}")
            return None

    def parse_spectral_data(self, data):
        """Parse spectral data from observation"""
        if not data or 'spectra' not in data:
            return None
        
        df = pd.DataFrame(data['spectra'])
        return df
    
    def plot_spectrum(self, spectral_data, title="Atmospheric Spectrum"):
        """Plot spectral data showing atmospheric composition"""
        if spectral_data is None or spectral_data.empty:
            print("No spectral data available to plot")
            return
        
        # Extract wavelength and flux data
        if 'wavelength' not in spectral_data.columns or 'flux' not in spectral_data.columns:
            print("Missing wavelength or flux data")
            return
            
        wavelengths = spectral_data['wavelength']
        flux = spectral_data['flux']
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, flux, 'b-')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Flux')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Mark key molecular features
        molecular_markers = {
            'H2O': [1.4, 1.9, 2.7],
            'CO2': [2.0, 4.3],
            'CH4': [1.6, 2.2, 3.3],
            'CO': [2.3, 4.6],
            'NH3': [1.5, 2.0, 3.0],
            'Na': [0.589],
            'K': [0.77]
        }
        
        for molecule, wavelengths_list in molecular_markers.items():
            for wl in wavelengths_list:
                if min(wavelengths) <= wl <= max(wavelengths):
                    plt.axvline(x=wl, color='r', linestyle='--', alpha=0.5)
                    plt.text(wl, max(flux) * 0.9, molecule, rotation=90, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_plot(self, spectral_data, title="Interactive Atmospheric Spectrum"):
        """Create an interactive spectrum plot using Plotly"""
        if spectral_data is None or spectral_data.empty:
            print("No spectral data available to plot")
            return
        
        # Extract wavelength and flux data
        if 'wavelength' not in spectral_data.columns or 'flux' not in spectral_data.columns:
            print("Missing wavelength or flux data")
            return
            
        wavelengths = spectral_data['wavelength']
        flux = spectral_data['flux']
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=flux,
            mode='lines',
            name='Spectrum'
        ))
        
        # Mark key molecular features
        molecular_markers = {
            'H2O': [1.4, 1.9, 2.7],
            'CO2': [2.0, 4.3],
            'CH4': [1.6, 2.2, 3.3],
            'CO': [2.3, 4.6],
            'NH3': [1.5, 2.0, 3.0],
            'Na': [0.589],
            'K': [0.77]
        }
        
        for molecule, wavelengths_list in molecular_markers.items():
            for wl in wavelengths_list:
                if min(wavelengths) <= wl <= max(wavelengths):
                    fig.add_vline(x=wl, line_width=1, line_dash="dash", line_color="red")
                    fig.add_annotation(
                        x=wl,
                        y=max(flux) * 0.9,
                        text=molecule,
                        showarrow=False,
                        textangle=90
                    )
        
        fig.update_layout(
            title=title,
            xaxis_title="Wavelength (μm)",
            yaxis_title="Flux",
            hovermode="closest"
        )
        
        fig.show()
    
    def analyze_atmospheric_composition(self, spectral_data):
        """
        Analyze atmospheric composition based on spectral data
        Returns dict with detected molecules and confidence levels
        """
        if spectral_data is None or spectral_data.empty:
            print("No spectral data available for analysis")
            return {}
        
        # This is a simplified analysis - in real life this would involve 
        # complex spectral modeling and absorption feature matching
        
        # Example wavelength ranges where we'd expect to see molecular features
        molecular_signatures = {
            'H2O': [(1.35, 1.45), (1.85, 1.95), (2.65, 2.75)],
            'CO2': [(1.95, 2.05), (4.25, 4.35)],
            'CH4': [(1.55, 1.65), (2.15, 2.25), (3.25, 3.35)],
            'CO': [(2.25, 2.35), (4.55, 4.65)],
            'NH3': [(1.45, 1.55), (1.95, 2.05), (2.95, 3.05)],
            'Na': [(0.585, 0.595)],
            'K': [(0.765, 0.775)]
        }
        
        # Extract wavelength and flux
        if 'wavelength' not in spectral_data.columns or 'flux' not in spectral_data.columns:
            print("Missing wavelength or flux data")
            return {}
            
        wavelengths = spectral_data['wavelength']
        flux = spectral_data['flux']
            
        # Convert to numpy arrays for easier processing
        wavelengths = np.array(wavelengths)
        flux = np.array(flux)
        
        # Track detected molecules and confidence
        detections = {}
        
        # For each molecule, check if we see absorption features at expected wavelengths
        for molecule, ranges in molecular_signatures.items():
            detection_strength = 0
            range_count = 0
            
            for wl_range in ranges:
                start, end = wl_range
                
                # Check if this wavelength range exists in our data
                if min(wavelengths) <= start and end <= max(wavelengths):
                    range_count += 1
                    
                    # Find indices in wavelength range
                    indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
                    
                    if len(indices) > 0:
                        # Calculate average flux in this range
                        avg_flux = np.mean(flux[indices])
                        
                        # Get average flux of nearby regions (for baseline)
                        before_indices = np.where((wavelengths >= start - 0.1) & (wavelengths < start))[0]
                        after_indices = np.where((wavelengths > end) & (wavelengths <= end + 0.1))[0]
                        
                        if len(before_indices) > 0 and len(after_indices) > 0:
                            baseline_flux = np.mean([
                                np.mean(flux[before_indices]),
                                np.mean(flux[after_indices])
                            ])
                            
                            # If flux in range is lower than baseline, might be absorption feature
                            if avg_flux < baseline_flux:
                                # Calculate strength of absorption relative to baseline
                                strength = (baseline_flux - avg_flux) / baseline_flux
                                detection_strength += strength
            
            # Calculate confidence based on detected strength and number of ranges checked
            if range_count > 0:
                confidence = (detection_strength / range_count) * 100
                
                # Only include molecules with some confidence level
                if confidence > 5:  # Arbitrary threshold
                    detections[molecule] = round(confidence, 1)
        
        return detections
    
    def plot_composition_bar(self, composition):
        """Plot bar chart of detected atmospheric composition"""
        if not composition:
            print("No composition data to plot")
            return
        
        molecules = list(composition.keys())
        confidences = list(composition.values())
        
        # Sort by confidence
        indices = np.argsort(confidences)
        molecules = [molecules[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(molecules, confidences, color='skyblue')
        plt.xlabel('Detection Confidence (%)')
        plt.ylabel('Molecule')
        plt.title('Atmospheric Composition Analysis')
        plt.xlim(0, 100)
        plt.grid(axis='x', alpha=0.3)
        
        # Add values at end of bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                     f'{width}%', va='center')
        
        plt.tight_layout()
        plt.show()


# Simple example of using the analyzer with mock data
def main():
    print("JWST Atmospheric Composition Analyzer")
    print("-------------------------------------")
    
    # Initialize analyzer
    # In a real application, you'd use your actual API key
    analyzer = JWSTDataAnalyzer(api_key=None)
    
    print("\nNote: This demo runs with mock data since an actual API key is required.")
    print("To use with real data, obtain an API key from https://jwstapi.com/\n")
    
    # Create mock spectral data for demonstration
    # This simulates data we might get from the API
    mock_wavelengths = np.linspace(0.5, 5.0, 500)  # wavelength in microns
    
    # Create a baseline flux
    mock_flux = 100 + 10 * np.sin(mock_wavelengths)
    
    # Add some absorption features for water, methane, and CO2
    for wl in [1.4, 1.9, 2.7]:  # Water features
        mock_flux -= 20 * np.exp(-((mock_wavelengths - wl) ** 2) / 0.01)
    
    for wl in [1.6, 3.3]:  # Methane features
        mock_flux -= 15 * np.exp(-((mock_wavelengths - wl) ** 2) / 0.008)
    
    for wl in [2.0, 4.3]:  # CO2 features
        mock_flux -= 25 * np.exp(-((mock_wavelengths - wl) ** 2) / 0.015)
    
    # Add noise
    mock_flux += np.random.normal(0, 2, size=len(mock_flux))
    
    # Create a mock spectral DataFrame
    mock_spectral_data = pd.DataFrame({
        'wavelength': mock_wavelengths,
        'flux': mock_flux
    })
    
    print("Analyzing mock data for exoplanet WASP-39b...")
    print("\nPlotting atmospheric spectrum...")
    analyzer.plot_spectrum(mock_spectral_data, title="WASP-39b Atmospheric Spectrum (Mock Data)")
    
    print("\nAnalyzing atmospheric composition...")
    composition = analyzer.analyze_atmospheric_composition(mock_spectral_data)
    
    print("\nDetected molecules and confidence levels:")
    for molecule, confidence in composition.items():
        print(f"- {molecule}: {confidence}% confidence")
    
    print("\nPlotting composition chart...")
    analyzer.plot_composition_bar(composition)
    
    print("\nCreating interactive spectrum visualization...")
    analyzer.interactive_plot(mock_spectral_data, title="WASP-39b Interactive Spectrum Analysis (Mock Data)")


if __name__ == "__main__":
    main()