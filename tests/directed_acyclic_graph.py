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


def test_get_root_nodes():
    """
    A -> B -> C
    """
    dag = DirectedAcyclicGraph()
    dag.add_edge(start='B', end='C')
    dag.add_edge(start='A', end='B')

    nodes = dag.get_root_nodes()
    assert nodes == ['A']
