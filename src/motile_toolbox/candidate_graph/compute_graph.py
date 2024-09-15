import logging
from typing import Any

import networkx as nx
import numpy as np

from .conflict_sets import compute_conflict_sets
from .utils import add_cand_edges, nodes_from_points_list, nodes_from_segmentation

logger = logging.getLogger(__name__)


def get_candidate_graph(
    segmentation: np.ndarray,
    max_edge_distance: float | None = None,
    num_nearest_neighbours: int | None = None,
    direction_candidate_graph: str = "backward",
    dT: int = 1,
) -> tuple[nx.DiGraph, list[set[Any]] | None]:
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance. If segmentation contains multiple hypotheses, will also
    return a list of conflicting node ids that cannot be selected together.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, h, [z], y, x), where h is the number of hypotheses.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        num_nearest_neighbours (int): Number of spatial neighbours to connect each
            segmentation to.
        dT (int): Edges will be built between frames which are separated by
            `dT` frames. If `dT ==1`, then frames `t` and `t+1` are connected. If
            `dT==2`, then frames `t` and `t+1` and frames `t` and `t+2` are
            connected.
        direction_candidate_graph (str): One of "forward" or "backward".
            Implies in which temporal direction, the candidate graph is built.

    Returns:
        tuple[nx.DiGraph, list[set[Any]] | None]: A candidate graph that can be passed
            to the motile solver, and a list of conflicting node ids.
    """
    num_hypotheses = segmentation.shape[1]

    # add nodes
    cand_graph, node_frame_dict = nodes_from_segmentation(segmentation)
    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        num_nearest_neighbours=num_nearest_neighbours,
        direction_candidate_graph=direction_candidate_graph,
        node_frame_dict=node_frame_dict,
        dT=dT,
    )

    logger.info(f"Candidate edges: {cand_graph.number_of_edges()}")

    # Compute conflict sets between segmentations
    conflicts = []
    if num_hypotheses > 1:
        for time, segs in enumerate(segmentation):
            conflicts.extend(compute_conflict_sets(segs, time))

    return cand_graph, conflicts


def get_candidate_graph_from_points_list(
    points_list: np.ndarray,
    max_edge_distance: float | None,
    num_nearest_neighbours: int | None,
    direction_candidate_graph: str = "backward",
    dT: int = 1,
) -> nx.DiGraph:
    """Construct a candidate graph from a points list.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order  (t, [z], y, x).
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.
        Multiple hypotheses not supported for points input.
    """
    # add nodes
    cand_graph, node_frame_dict = nodes_from_points_list(points_list)
    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")
    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        num_nearest_neighbours=num_nearest_neighbours,
        direction_candidate_graph=direction_candidate_graph,
        node_frame_dict=node_frame_dict,
        dT=dT,
    )
    return cand_graph
