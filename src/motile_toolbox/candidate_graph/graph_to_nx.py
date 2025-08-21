import networkx as nx
from motile import TrackGraph


def graph_to_nx(graph: TrackGraph) -> nx.DiGraph:
    """Convert a motile TrackGraph into a networkx DiGraph.

    Args:
        graph (TrackGraph): TrackGraph to be converted to networkx

    Returns:
        nx.DiGraph: Directed networkx graph with same nodes, edges, and attributes.
    """
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(graph.nodes.items())

    edges_list = []
    for edge_id, _ in graph.edges.items():
        src, dst = edge_id

        # Make sure both src and dst are iterable tuples/lists of nodes
        if not isinstance(src, tuple):
            src = (src,)
        if not isinstance(dst, tuple):
            dst = (dst,)

        # Connect all source nodes to all destination nodes
        for s in src:
            for d in dst:
                edges_list.append((s, d))

    nx_graph.add_edges_from(edges_list)
    return nx_graph
