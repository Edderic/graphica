"""
DirectedAcyclicGraph class
"""


class DirectedAcyclicGraph:
    """
    Directed Acyclic Graph. Useful for representing Bayesian Networks.
    """
    def __init__(self):
        self.children = {}
        self.nodes = {}

    def add_edge(self, start, end):
        """
        Add edge from start node to end node.

        Parameters:
            start: str
                Start node.
            end: str
                End node.
        """

        if start not in self.children:
            self.children[start] = [end]
        elif end not in self.children[start]:
            self.children[start].append(end)

    def add_node(self, node):
        """
        Add edge from start node to end node.

        Parameters:
            node: str
                Node to add.
        """

        if node not in self.nodes:
            self.nodes[node] = 1

    def get_neighbors(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """

        return self.get_parents(node) + self.get_children(node)

    def get_parents(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """
        parents = []

        for other_node, children in self.children.items():
            if node in children:
                parents.append(other_node)

        return parents

    def get_children(self, node):
        """
        Parameters:
            node: str

        Returns: list[str]
        """

        if node not in self.children:
            return []

        return self.children[node]


