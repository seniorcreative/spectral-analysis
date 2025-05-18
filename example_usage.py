from jwst_analyzer import JWSTDataAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demo_wasp39b():
    """
    Demonstrate analysis on mock data for WASP-39b
    This exoplanet is one of the best characterized by JWST
    """
    print("Analyzing WASP-39b (Hot Saturn)")
    print("-" * 40)
    
    # Create analyzer without API key (mock data mode)
    analyzer = JWSTDataAnalyzer()
    
    # Create realistic mock data for WASP-39b based on actual JWST findings
    # WASP-39b has confirmed detections of H2O, CO2, CO, Na, and SO2
    wavelengths = np.linspace(0.5, 5.0, 500)
    
    # Create baseline flux
    flux = 100 + 5 * np.sin(wavelengths * 0.5)
    
    # Add absorption features for confirmed molecules
    # Water (strong features)
    for wl in [1.4, 1.9, 2.7]:
        flux -= 30 * np.exp(-((wavelengths - wl) ** 2) / 0.01)
    
    # CO2 (strong detection)
    for wl in [2.0, 4.3]:
        flux -= 35 * np.exp(-((wavelengths - wl) ** 2) / 0.012)
    
    # CO (moderate detection)
    for wl in [2.3, 4.6]:
        flux -= 15 * np.exp(-((wavelengths - wl) ** 2) / 0.01)
    
    # Na (weak detection)
    flux -= 10 * np.exp(-((wavelengths - 0.589) ** 2) / 0.005)
    
    # SO2 (special feature of WASP-39b discovered by JWST)
    flux -= 18 * np.exp(-((wavelengths - 4.0) ** 2) / 0.015)
    
    # Add noise
    flux += np.random.normal(0, 2, size=len(flux))
    
    # Create spectral data
    spectral_data = pd.DataFrame({
        'wavelength': wavelengths,
        'flux': flux
    })
    
    # Analyze and visualize
    print("Generating spectrum plot...")
    analyzer.plot_spectrum(spectral_data, title="WASP-39b Atmospheric Spectrum (Mock Data)")
    
    print("\nAnalyzing atmospheric composition...")
    composition = analyzer.analyze_atmospheric_composition(spectral_data)
    
    print("\nDetected molecules:")
    for molecule, confidence in composition.items():
        print(f"- {molecule}: {confidence}% confidence")
    
    print("\nPlotting composition chart...")
    analyzer.plot_composition_bar(composition)
    
    print("\nComparison with actual JWST findings:")
    print("- JWST detected H2O, CO2, CO, Na, and SO2 in WASP-39b")
    print("- First detection of sulfur dioxide in an exoplanet atmosphere")
    print("- Suggests photochemistry (reactions triggered by starlight)")
    
    # Generate interactive plot
    print("\nCreating interactive visualization...")
    analyzer.interactive_plot(spectral_data, title="WASP-39b Atmospheric Analysis")


def demo_wasp96b():
    """
    Demonstrate analysis on mock data for WASP-96b
    Another exoplanet studied by JWST with water vapor detection
    """
    print("\n\nAnalyzing WASP-96b (Hot Gas Giant)")
    print("-" * 40)
    
    # Create analyzer without API key (mock data mode)
    analyzer = JWSTDataAnalyzer()
    
    # Create mock data for WASP-96b
    # Known for its water vapor detection
    wavelengths = np.linspace(0.5, 5.0, 500)
    
    # Create baseline flux
    flux = 100 + 7 * np.sin(wavelengths * 0.3)
    
    # Add water features (strong detection)
    for wl in [1.4, 1.9, 2.7]:
        flux -= 25 * np.exp(-((wavelengths - wl) ** 2) / 0.01)
    
    # Add some evidence of clouds (dampened spectral features)
    flux = flux * (0.9 + 0.1 * np.sin(wavelengths * 10))
    
    # Add noise
    flux += np.random.normal(0, 3, size=len(flux))
    
    # Create spectral data
    spectral_data = pd.DataFrame({
        'wavelength': wavelengths,
        'flux': flux
    })
    
    # Analyze and visualize
    print("Generating spectrum plot...")
    analyzer.plot_spectrum(spectral_data, title="WASP-96b Atmospheric Spectrum (Mock Data)")
    
    print("\nAnalyzing atmospheric composition...")
    composition = analyzer.analyze_atmospheric_composition(spectral_data)
    
    print("\nDetected molecules:")
    for molecule, confidence in composition.items():
        print(f"- {molecule}: {confidence}% confidence")
    
    print("\nPlotting composition chart...")
    analyzer.plot_composition_bar(composition)
    
    print("\nComparison with actual JWST findings:")
    print("- JWST detected clear signatures of water vapor")
    print("- Evidence of clouds and haze that were missed by previous observations")
    print("- Water features less pronounced than in some model predictions")
    
    # Generate interactive plot
    print("\nCreating interactive visualization...")
    analyzer.interactive_plot(spectral_data, title="WASP-96b Atmospheric Analysis")


def compare_planets():
    """Compare atmospheric data from multiple planets"""
    print("\n\nComparing Exoplanet Atmospheres")
    print("-" * 40)
    
    # For a real implementation, this would fetch data for multiple planets
    # and compare their compositions
    
    # Simple mock comparison
    planets = ['WASP-39b', 'WASP-96b', 'WASP-43b', 'HD 189733b']
    
    # Create mock detection data for multiple molecules across planets
    molecules = ['H2O', 'CO2', 'CH4', 'CO', 'Na', 'K', 'NH3']
    
    # Mock confidence values for each planet-molecule combination
    data = {
        'WASP-39b': {
            'H2O': 95.2,
            'CO2': 92.7,
            'CH4': 12.3,
            'CO': 78.1,
            'Na': 65.4,
            'K': 23.1,
            'NH3': 8.5
        },
        'WASP-96b': {
            'H2O': 85.7,
            'CO2': 45.2,
            'CH4': 5.8,
            'CO': 31.2,
            'Na': 42.5,
            'K': 16.4,
            'NH3': 3.1
        },
        'WASP-43b': {
            'H2O': 76.3,
            'CO2': 82.5,
            'CH4': 44.7,
            'CO': 64.2,
            'Na': 22.3,
            'K': 18.9,
            'NH3': 26.7
        },
        'HD 189733b': {
            'H2O': 89.3,
            'CO2': 62.1,
            'CH4': 75.6,
            'CO': 36.8,
            'Na': 74.2,
            'K': 53.7,
            'NH3': 4.3
        }
    }
    
    # Create a heatmap of molecular detections across planets
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(df.values, cmap='YlOrRd')
    plt.colorbar(label='Detection Confidence (%)')
    plt.xticks(np.arange(len(planets)), planets, rotation=45)
    plt.yticks(np.arange(len(molecules)), molecules)
    plt.title('Comparative Atmospheric Composition of Exoplanets')
    
    # Add text annotations to cells
    for i in range(len(molecules)):
        for j in range(len(planets)):
            plt.text(j, i, f"{df.iloc[i, j]:.1f}%", 
                     ha="center", va="center", 
                     color="black" if df.iloc[i, j] < 70 else "white")
    
    plt.tight_layout()
    plt.show()
    
    print("\nComparison Insights:")
    print("- Hot Jupiters and Hot Saturns show different molecular compositions")
    print("- Water is commonly detected across multiple planets")
    print("- Methane abundance varies significantly between planets")
    print("- Atmospheric chemistry depends on temperature, UV radiation, and planet mass")


if __name__ == "__main__":
    print("JWST Atmospheric Composition Analysis Examples")
    print("=============================================")
    print("Note: Running with mock data. For real analysis, obtain an API key from jwstapi.com")
    
    # Run the example analyses
    demo_wasp39b()
    demo_wasp96b()
    compare_planets()
    
    print("\nFurther Analysis:")
    print("To perform analysis on real JWST data, obtain an API key and modify the examples")
    print("to use the actual API endpoints instead of mock data.")