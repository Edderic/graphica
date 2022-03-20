from ..linx.directed_acyclic_graph import DirectedAcyclicGraph


def test_topological_sort_chain():
    """
    A -> B -> C
    """
    dag = DirectedAcyclicGraph()
    dag.add_edge(start='B', end='C')
    dag.add_edge(start='A', end='B')

    sorted_nodes = dag.topological_sort()
    assert sorted_nodes == ['A', 'B', 'C']


def test_topological_sort_complex():
    """
         X -> D
         |    |
         v    v
    A -> B -> C

    """
    dag = DirectedAcyclicGraph()

    dag.add_edge(start='D', end='C')
    dag.add_edge(start='B', end='C')
    dag.add_edge(start='X', end='B')
    dag.add_edge(start='A', end='B')
    dag.add_edge(start='X', end='D')

    sorted_nodes = dag.topological_sort()
    acceptable_1 = sorted_nodes == ['X', 'A', 'D', 'B', 'C']
    acceptable_2 = sorted_nodes == ['X', 'A', 'B', 'D', 'C']
    acceptable_3 = sorted_nodes == ['A', 'X', 'D', 'B', 'C']
    acceptable_4 = sorted_nodes == ['A', 'X', 'B', 'D', 'C']
    assert acceptable_1 or acceptable_2 or acceptable_3 or acceptable_4


def test_get_root_nodes():
    """
    A -> B -> C
    """
    dag = DirectedAcyclicGraph()
    dag.add_edge(start='B', end='C')
    dag.add_edge(start='A', end='B')

    nodes = dag.get_root_nodes()
    assert nodes == ['A']
