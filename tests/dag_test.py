from ..graphica.ds import DirectedAcyclicGraph as DAG


def test_get_neighbors():
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    assert ["B"] == dag.get_neighbors("A")
    assert ["A", "C"] == dag.get_neighbors("B")
    assert ["B"] == dag.get_neighbors("C")


def test_get_parents():
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    assert [] == dag.get_parents("A")
    assert ["A"] == dag.get_parents("B")
    assert ["B"] == dag.get_parents("C")


def test_get_children():
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")

    assert ["B"] == dag.get_children("A")
    assert ["C"] == dag.get_children("B")
    assert [] == dag.get_children("C")
