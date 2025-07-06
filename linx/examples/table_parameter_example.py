"""
Example demonstrating the new table parameter for ConditionalProbabilityTable.
"""
from ..ds import BayesianNetwork as BN, ConditionalProbabilityTable as CPT


def create_weather_network_with_table():
    """
    Create a simple weather Bayesian Network using the table parameter:
    
    Rain -> WetGrass
    Sprinkler -> WetGrass
    Rain -> Cloudy
    Cloudy -> Sprinkler
    """
    
    # Create the network
    bayesian_network = BN()
    
    # Prior for Rain using table parameter
    cpt_rain = CPT(
        table=[
            {'Rain': 0, 'value': 0.8},
            {'Rain': 1, 'value': 0.2}
        ],
        outcomes=['Rain']
    )
    
    bayesian_network.add_node(cpt_rain)
    
    # Prior for Cloudy using table parameter
    cpt_cloudy = CPT(
        table=[
            {'Cloudy': 0, 'value': 0.5},
            {'Cloudy': 1, 'value': 0.5}
        ],
        outcomes=['Cloudy']
    )
    
    bayesian_network.add_node(cpt_cloudy)
    
    # Sprinkler depends on Cloudy using table parameter
    cpt_sprinkler = CPT(
        table=[
            {'Cloudy': 0, 'Sprinkler': 0, 'value': 0.5},
            {'Cloudy': 0, 'Sprinkler': 1, 'value': 0.5},
            {'Cloudy': 1, 'Sprinkler': 0, 'value': 0.9},
            {'Cloudy': 1, 'Sprinkler': 1, 'value': 0.1}
        ],
        outcomes=['Sprinkler'],
        givens=['Cloudy']
    )
    
    bayesian_network.add_edge(cpt_sprinkler)
    
    # WetGrass depends on Rain and Sprinkler using table parameter
    cpt_wetgrass = CPT(
        table=[
            {'Rain': 0, 'Sprinkler': 0, 'WetGrass': 0, 'value': 1.0},
            {'Rain': 0, 'Sprinkler': 0, 'WetGrass': 1, 'value': 0.0},
            {'Rain': 0, 'Sprinkler': 1, 'WetGrass': 0, 'value': 0.1},
            {'Rain': 0, 'Sprinkler': 1, 'WetGrass': 1, 'value': 0.9},
            {'Rain': 1, 'Sprinkler': 0, 'WetGrass': 0, 'value': 0.1},
            {'Rain': 1, 'Sprinkler': 0, 'WetGrass': 1, 'value': 0.9},
            {'Rain': 1, 'Sprinkler': 1, 'WetGrass': 0, 'value': 0.01},
            {'Rain': 1, 'Sprinkler': 1, 'WetGrass': 1, 'value': 0.99}
        ],
        outcomes=['WetGrass'],
        givens=['Rain', 'Sprinkler']
    )
    
    bayesian_network.add_edge(cpt_wetgrass)
    
    return bayesian_network


def run_table_parameter_example():
    """
    Run the table parameter example.
    """
    print("Creating weather Bayesian Network using table parameter...")
    network = create_weather_network_with_table()
    
    print("\nGenerating 5 samples:")
    print("-" * 50)
    
    for i in range(5):
        particle = network.sample()
        print(f"Sample {i+1}:")
        print(f"  Rain: {particle.get_value('Rain')}")
        print(f"  Cloudy: {particle.get_value('Cloudy')}")
        print(f"  Sprinkler: {particle.get_value('Sprinkler')}")
        print(f"  WetGrass: {particle.get_value('WetGrass')}")
        print()
    
    print("Table parameter example completed!")


if __name__ == "__main__":
    run_table_parameter_example() 