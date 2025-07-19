"""
Example demonstrating forward sampling from a Bayesian Network.
"""

import pandas as pd
from ..data import ParquetData
from ..ds import BayesianNetwork as BN, ConditionalProbabilityTable as CPT
from ..misc import get_tmp_path


def create_weather_network():
    """
    Create a simple weather Bayesian Network:

    Rain -> WetGrass
    Sprinkler -> WetGrass
    Rain -> Cloudy
    Cloudy -> Sprinkler
    """

    # Create the network
    bayesian_network = BN()

    # Prior for Rain
    df_rain = pd.DataFrame([{"Rain": 0, "value": 0.8}, {"Rain": 1, "value": 0.2}])

    cpt_rain = CPT(
        ParquetData(df_rain, storage_folder=get_tmp_path()), outcomes=["Rain"]
    )

    bayesian_network.add_node(cpt_rain)

    # Prior for Cloudy
    df_cloudy = pd.DataFrame([{"Cloudy": 0, "value": 0.5}, {"Cloudy": 1, "value": 0.5}])

    cpt_cloudy = CPT(
        ParquetData(df_cloudy, storage_folder=get_tmp_path()), outcomes=["Cloudy"]
    )

    bayesian_network.add_node(cpt_cloudy)

    # Sprinkler depends on Cloudy
    df_sprinkler = pd.DataFrame(
        [
            {"Cloudy": 0, "Sprinkler": 0, "value": 0.5},
            {"Cloudy": 0, "Sprinkler": 1, "value": 0.5},
            {"Cloudy": 1, "Sprinkler": 0, "value": 0.9},
            {"Cloudy": 1, "Sprinkler": 1, "value": 0.1},
        ]
    )

    cpt_sprinkler = CPT(
        ParquetData(df_sprinkler, storage_folder=get_tmp_path()),
        outcomes=["Sprinkler"],
        givens=["Cloudy"],
    )

    bayesian_network.add_node(cpt_sprinkler)

    # WetGrass depends on Rain and Sprinkler
    df_wetgrass = pd.DataFrame(
        [
            {"Rain": 0, "Sprinkler": 0, "WetGrass": 0, "value": 1.0},
            {"Rain": 0, "Sprinkler": 0, "WetGrass": 1, "value": 0.0},
            {"Rain": 0, "Sprinkler": 1, "WetGrass": 0, "value": 0.1},
            {"Rain": 0, "Sprinkler": 1, "WetGrass": 1, "value": 0.9},
            {"Rain": 1, "Sprinkler": 0, "WetGrass": 0, "value": 0.1},
            {"Rain": 1, "Sprinkler": 0, "WetGrass": 1, "value": 0.9},
            {"Rain": 1, "Sprinkler": 1, "WetGrass": 0, "value": 0.01},
            {"Rain": 1, "Sprinkler": 1, "WetGrass": 1, "value": 0.99},
        ]
    )

    cpt_wetgrass = CPT(
        ParquetData(df_wetgrass, storage_folder=get_tmp_path()),
        outcomes=["WetGrass"],
        givens=["Rain", "Sprinkler"],
    )

    bayesian_network.add_node(cpt_wetgrass)

    return bayesian_network


def run_sampling_example():
    """
    Run the sampling example.
    """
    print("Creating weather Bayesian Network...")
    network = create_weather_network()

    print("\nGenerating 10 samples:")
    print("-" * 50)

    for i in range(10):
        particle = network.sample()
        print(f"Sample {i+1}:")
        print(f"  Rain: {particle.get_value('Rain')}")
        print(f"  Cloudy: {particle.get_value('Cloudy')}")
        print(f"  Sprinkler: {particle.get_value('Sprinkler')}")
        print(f"  WetGrass: {particle.get_value('WetGrass')}")
        print()

    print("Sampling completed!")


if __name__ == "__main__":
    run_sampling_example()
