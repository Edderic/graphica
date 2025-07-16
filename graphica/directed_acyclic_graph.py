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

        if node not in self.children:
            self.children[node] = []

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

    def get_root_nodes(self):
        """
        Get the root nodes.

        Returns: list[str]
        """
        root_nodes = []

        for node in self.children:
            node_is_a_child = False

            for _, child_nodes in self.children.items():
                if node in child_nodes:
                    node_is_a_child = True
                    break

            if not node_is_a_child:
                root_nodes.append(node)

        return root_nodes

    def topological_sort(self):
        """
        Get a sorted list of variables.
        """
        visited = {}
        roots = self.get_root_nodes()
        sorted_vars = []

        for var in roots:
            visited[var] = 1
            sorted_vars.append(var)

        while len(set(self.get_nodes()) - set(visited.keys())) > 0:
            unvisiteds = list(set(self.get_nodes()) - set(visited.keys()))

            for unvisited in unvisiteds:
                parents = self.get_parents(unvisited)
                if set(parents) - set(visited.keys()) == set({}):
                    sorted_vars.append(unvisited)
                    visited[unvisited] = 1

        return sorted_vars

    def get_nodes(self):
        """
        Get all the nodes in the DAG.
        """
        nodes = set({})

        for parent, children in self.children.items():
            nodes = nodes.union(set([parent]).union(children))

        return list(nodes)
