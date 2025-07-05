from ..linx.directed_acyclic_graph import DirectedAcyclicGraph


def test_add_edge():
    """Test adding edges to the DAG."""
    dag = DirectedAcyclicGraph()
    
    # Test adding first edge
    dag.add_edge('A', 'B')
    assert dag.children == {'A': ['B']}
    
    # Test adding second edge from same parent
    dag.add_edge('A', 'C')
    assert dag.children == {'A': ['B', 'C']}
    
    # Test adding edge with new parent
    dag.add_edge('B', 'D')
    assert dag.children == {'A': ['B', 'C'], 'B': ['D']}
    
    # Test adding duplicate edge (should not add)
    dag.add_edge('A', 'B')
    assert dag.children == {'A': ['B', 'C'], 'B': ['D']}


def test_add_node():
    """Test adding nodes to the DAG."""
    dag = DirectedAcyclicGraph()
    
    # Test adding first node
    dag.add_node('A')
    assert dag.children == {'A': []}
    
    # Test adding second node
    dag.add_node('B')
    assert dag.children == {'A': [], 'B': []}
    
    # Test adding node that already exists
    dag.add_node('A')
    assert dag.children == {'A': [], 'B': []}


def test_get_children():
    """Test getting children of nodes."""
    dag = DirectedAcyclicGraph()
    dag.add_edge('A', 'B')
    dag.add_edge('A', 'C')
    dag.add_edge('B', 'D')
    
    # Test getting children of existing node
    assert dag.get_children('A') == ['B', 'C']
    assert dag.get_children('B') == ['D']
    
    # Test getting children of node with no children
    assert dag.get_children('C') == []
    
    # Test getting children of non-existent node
    assert dag.get_children('X') == []


def test_get_parents():
    """Test getting parents of nodes."""
    dag = DirectedAcyclicGraph()
    dag.add_edge('A', 'B')
    dag.add_edge('C', 'B')
    dag.add_edge('B', 'D')
    
    # Test getting parents of node with multiple parents
    parents = dag.get_parents('B')
    assert set(parents) == {'A', 'C'}
    
    # Test getting parents of node with single parent
    assert dag.get_parents('D') == ['B']
    
    # Test getting parents of root node
    assert dag.get_parents('A') == []
    assert dag.get_parents('C') == []
    
    # Test getting parents of non-existent node
    assert dag.get_parents('X') == []


def test_get_neighbors():
    """Test getting all neighbors (parents + children) of nodes."""
    dag = DirectedAcyclicGraph()
    dag.add_edge('A', 'B')
    dag.add_edge('C', 'B')
    dag.add_edge('B', 'D')
    
    # Test getting neighbors of middle node
    neighbors = dag.get_neighbors('B')
    assert set(neighbors) == {'A', 'C', 'D'}
    
    # Test getting neighbors of root node
    neighbors = dag.get_neighbors('A')
    assert set(neighbors) == {'B'}
    
    # Test getting neighbors of leaf node
    neighbors = dag.get_neighbors('D')
    assert set(neighbors) == {'B'}
    
    # Test getting neighbors of non-existent node
    assert dag.get_neighbors('X') == []


def test_get_nodes():
    """Test getting all nodes in the DAG."""
    dag = DirectedAcyclicGraph()
    
    # Test empty DAG
    assert dag.get_nodes() == []
    
    # Test DAG with nodes but no edges
    dag.add_node('A')
    dag.add_node('B')
    nodes = dag.get_nodes()
    assert set(nodes) == {'A', 'B'}
    
    # Test DAG with edges
    dag.add_edge('A', 'C')
    dag.add_edge('B', 'D')
    nodes = dag.get_nodes()
    assert set(nodes) == {'A', 'B', 'C', 'D'}


def test_get_root_nodes():
    """Test getting root nodes (nodes with no parents)."""
    dag = DirectedAcyclicGraph()
    
    # Test empty DAG
    assert dag.get_root_nodes() == []
    
    # Test simple chain
    dag.add_edge('A', 'B')
    dag.add_edge('B', 'C')
    roots = dag.get_root_nodes()
    assert roots == ['A']
    
    # Test multiple roots
    dag.add_edge('X', 'B')
    dag.add_edge('Y', 'C')
    roots = dag.get_root_nodes()
    assert set(roots) == {'A', 'X', 'Y'}
    
    # Test isolated nodes
    dag.add_node('Z')
    roots = dag.get_root_nodes()
    assert set(roots) == {'A', 'X', 'Y', 'Z'}


def test_get_root_nodes_complex():
    """
    Test root nodes in a more complex graph:
         X -> D
         |    |
         v    v
    A -> B -> C
    """
    dag = DirectedAcyclicGraph()
    dag.add_edge('D', 'C')
    dag.add_edge('B', 'C')
    dag.add_edge('X', 'B')
    dag.add_edge('A', 'B')
    dag.add_edge('X', 'D')
    
    roots = dag.get_root_nodes()
    assert set(roots) == {'A', 'X'}


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


def test_topological_sort_empty():
    """Test topological sort on empty DAG."""
    dag = DirectedAcyclicGraph()
    sorted_nodes = dag.topological_sort()
    assert sorted_nodes == []


def test_topological_sort_single_node():
    """Test topological sort on DAG with single node."""
    dag = DirectedAcyclicGraph()
    dag.add_node('A')
    sorted_nodes = dag.topological_sort()
    assert sorted_nodes == ['A']


def test_topological_sort_disconnected():
    """Test topological sort on DAG with disconnected components."""
    dag = DirectedAcyclicGraph()
    dag.add_edge('A', 'B')
    dag.add_edge('C', 'D')
    dag.add_node('E')
    
    sorted_nodes = dag.topological_sort()
    # All valid topological orders should contain all nodes
    assert set(sorted_nodes) == {'A', 'B', 'C', 'D', 'E'}
    # Check that dependencies are respected
    a_idx = sorted_nodes.index('A')
    b_idx = sorted_nodes.index('B')
    c_idx = sorted_nodes.index('C')
    d_idx = sorted_nodes.index('D')
    assert a_idx < b_idx
    assert c_idx < d_idx


def test_edge_cases():
    """Test various edge cases."""
    dag = DirectedAcyclicGraph()
    
    # Test adding edge to non-existent nodes
    dag.add_edge('A', 'B')
    assert dag.children == {'A': ['B']}
    
    # Test adding node after edge
    dag.add_node('C')
    assert dag.children == {'A': ['B'], 'C': []}
    
    # Test getting relationships for non-existent nodes
    assert dag.get_children('X') == []
    assert dag.get_parents('X') == []
    assert dag.get_neighbors('X') == []
    
    # Test with isolated nodes
    dag.add_node('D')
    nodes = dag.get_nodes()
    assert set(nodes) == {'A', 'B', 'C', 'D'}
    
    roots = dag.get_root_nodes()
    assert set(roots) == {'A', 'C', 'D'}


def test_multiple_edges_same_direction():
    """Test adding multiple edges in the same direction."""
    dag = DirectedAcyclicGraph()
    dag.add_edge('A', 'B')
    dag.add_edge('A', 'B')  # Duplicate edge
    dag.add_edge('A', 'C')
    dag.add_edge('B', 'C')
    
    assert dag.get_children('A') == ['B', 'C']
    assert dag.get_parents('C') == ['A', 'B']
    assert dag.get_parents('B') == ['A']
