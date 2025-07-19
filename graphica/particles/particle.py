"""
Particle class for representing sampled values from Bayesian Networks.
"""


class Particle:
    """
    A particle represents a single sample from a Bayesian Network.

    A particle stores the values of all variables in the network for one sample.

    Parameters:
        values: dict
            Dictionary where keys are variable names and values are the sampled
            values for those variables.
    """

    def __init__(self, values=None):
        if values is None:
            self.values = {}
        else:
            self.values = values.copy()

        # Track acceptance/rejection status
        self.status = None

    def __repr__(self):
        return f"Particle({self.values})"

    def __str__(self):
        return f"Particle({self.values})"

    def get_value(self, variable):
        """
        Get the sampled value for a specific variable.

        Parameters:
            variable: str
                Name of the variable.

        Returns:
            The sampled value for the variable.

        Raises:
            KeyError: If the variable is not in this particle.
        """
        return self.values[variable]

    def set_value(self, variable, value):
        """
        Set the sampled value for a specific variable.

        Parameters:
            variable: str
                Name of the variable.
            value:
                The sampled value for the variable.
        """
        self.values[variable] = value

    def get_all_values(self):
        """
        Get all sampled values as a dictionary.

        Returns:
            dict: Dictionary mapping variable names to their sampled values.
        """
        return self.values.copy()

    def has_variable(self, variable):
        """
        Check if this particle has a value for the given variable.

        Parameters:
            variable: str
                Name of the variable.

        Returns:
            bool: True if the variable exists in this particle, False otherwise.
        """
        return variable in self.values

    def get_variables(self):
        """
        Get all variable names in this particle.

        Returns:
            list: List of variable names.
        """
        return list(self.values.keys())

    def copy(self):
        """
        Create a deep copy of this particle.

        Returns:
            Particle: A new particle with the same values and status.
        """
        new_particle = Particle(self.values.copy())
        new_particle.status = self.get_status()
        return new_particle

    def accept(self):
        """
        Mark this particle as accepted.
        """
        self.status = "accepted"

    def reject(self):
        """
        Mark this particle as rejected.
        """
        self.status = "rejected"

    def get_status(self):
        """
        Get the current status of this particle.

        Returns:
            str: The status of the particle ("accepted", "rejected", or None).
        """
        return self.status

    def is_accepted(self):
        """
        Check if this particle was accepted.

        Returns:
            bool: True if the particle was accepted, False otherwise.
        """
        return self.get_status() == "accepted"

    def is_rejected(self):
        """
        Check if this particle was rejected.

        Returns:
            bool: True if the particle was rejected, False otherwise.
        """
        return self.get_status() == "rejected"
