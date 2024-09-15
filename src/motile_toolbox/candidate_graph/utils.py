import logging
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from skimage.measure import regionprops
from tqdm import tqdm

from .graph_attributes import NodeAttr

logger = logging.getLogger(__name__)


def get_node_id(time: int, label_id: int, hypothesis_id: int | None = None) -> str:
    """Construct a node id given the time frame, segmentation label id, and
    optionally the hypothesis id. This function is not designed for candidate graphs
    that do not come from segmentations, but could be used if there is a similar
    "detection id" that is unique for all cells detected in a given frame.

    Args:
        time (int): The time frame the node is in
        label_id (int): The label the node has in the segmentation.
        hypothesis_id (int | None, optional): An integer representing which hypothesis
            the segmentation came from, if applicable. Defaults to None.

    Returns:
        str: A string to use as the node id in the candidate graph. Assuming that label
        ids are not repeated in the same time frame and hypothesis, it is unique.
    """
    if hypothesis_id is not None:
        return f"{time}_{hypothesis_id}_{label_id}"
    else:
        return f"{time}_{label_id}"


def nodes_from_segmentation(
    segmentation: np.ndarray,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, h, [z], y, x), where h is the number of hypotheses.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}
    print("Extracting nodes from segmentation")
    num_hypotheses = segmentation.shape[1]

    for t in tqdm(range(len(segmentation))):
        if t == 0:
            segs_tm1 = np.zeros_like(segmentation[t])
        else:
            segs_tm1 = segmentation[t - 1]

        segs = segmentation[t]  # C H W

        if t == len(segmentation) - 1:
            segs_tp1 = np.zeros_like(segmentation[t])
        else:
            segs_tp1 = segmentation[t + 1]

        hypo_id: int | None

        for hypo_id, hypo in enumerate(segs):
            if num_hypotheses == 1:
                hypo_id = None
            nodes_in_frame = []
            props = regionprops(hypo)
            for regionprop in props:
                node_id = get_node_id(t, regionprop.label, hypothesis_id=hypo_id)
                attrs = {
                    NodeAttr.TIME.value: t,
                }
                if np.sum(segs) > 0 and np.sum(segs_tp1) == 0:  # disappearance
                    attrs[NodeAttr.IGNORE_DISAPPEAR_COST.value] = True
                if np.sum(segs_tm1) == 0 and np.sum(segs) > 0:  # appearance
                    attrs[NodeAttr.IGNORE_APPEAR_COST.value] = True

                attrs[NodeAttr.SEG_ID.value] = regionprop.label
                if hypo_id is not None:
                    attrs[NodeAttr.SEG_HYPO.value] = hypo_id
                centroid = regionprop.centroid  # [z,] y, x
                attrs[NodeAttr.POS.value] = centroid
                cand_graph.add_node(node_id, **attrs)
                nodes_in_frame.append(node_id)
            if nodes_in_frame:
                if t not in node_frame_dict:
                    node_frame_dict[t] = []
                node_frame_dict[t].extend(nodes_in_frame)

    return cand_graph, node_frame_dict


def nodes_from_points_list(
    points_list: np.ndarray,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a list of points. Uses the index of the
    point in the list as its unique id.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order (t, [z], y, x).

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}
    print("Extracting nodes from points list")
    t_min = int(np.min(points_list[:, 1]))
    t_max = int(np.max(points_list[:, 1]))
    for i, point in enumerate(points_list):
        # assume seg_id, t, [z], y, x, p_id
        id_ = int(point[0])
        t = int(point[1])
        pos = list(point[2:-1].astype(int))
        node_id = str(t) + "_" + str(id_)  # t_id
        attrs = {
            NodeAttr.TIME.value: t,
            NodeAttr.POS.value: pos,
        }
        if t == t_min:
            attrs[NodeAttr.IGNORE_APPEAR_COST.value] = True
        if t == t_max:
            attrs[NodeAttr.IGNORE_DISAPPEAR_COST.value] = True

        attrs[NodeAttr.SEG_ID.value] = id_
        attrs[NodeAttr.PINNED.value] = True
        cand_graph.add_node(node_id, **attrs)

        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node_id)
    return cand_graph, node_frame_dict


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data[NodeAttr.TIME.value]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> KDTree:
    """Builds a kd-tree from available positions of segmentations at a given time
    frame.
    """
    positions = [cand_graph.nodes[node][NodeAttr.POS.value] for node in node_ids]
    return KDTree(positions), positions


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    num_nearest_neighbours: int,
    direction_candidate_graph: str,
    node_frame_dict: None | dict[int, list[Any]] = None,
    dT: int = 1,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        num_nearest_neighbours (int): Each segmentation is connected to its
            `num_nearest_neighbours` spatial neighbours in the next frame.
        direction_candidate_graph (str): One of "forward" or "backward".
            Indicates the temporal direction.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
        dT (int): If `dT==1`, then edges are constructed between frames `t` and
            `t+1`. If `dT==2`, then edges are constructed between frames `t`
            and `t+1`, and also `t` and `t+2`. And so on. Allows for including
            skip edges.
    """
    print("Extracting candidate edges")
    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph)
    if direction_candidate_graph == "forward":
        frames = sorted(node_frame_dict.keys())
    elif direction_candidate_graph == "backward":
        frames = sorted(node_frame_dict.keys(), reverse=True)

    for frame in tqdm(frames):
        prev_node_ids = node_frame_dict[frame]
        prev_kdtree, prev_positions = create_kdtree(cand_graph, prev_node_ids)
        if direction_candidate_graph == "forward":
            for t_next in range(frame + 1, frame + dT + 1):
                if t_next not in node_frame_dict:
                    continue
                next_node_ids = node_frame_dict[t_next]
                next_kdtree, next_positions = create_kdtree(cand_graph, next_node_ids)
                if num_nearest_neighbours is not None:
                    _, matched_indices = next_kdtree.query(
                        x=prev_positions, k=num_nearest_neighbours
                    )
                elif max_edge_distance is not None:
                    matched_indices = prev_kdtree.query_ball_tree(
                        next_kdtree, max_edge_distance
                    )

                for prev_node_id, next_node_indices in zip(
                    prev_node_ids, matched_indices
                ):
                    for next_node_index in next_node_indices:
                        next_node_id = next_node_ids[next_node_index]
                        cand_graph.add_edge(prev_node_id, next_node_id)
        elif direction_candidate_graph == "backward":
            for t_next in range(frame - 1, frame - dT - 1, -1):
                if t_next not in node_frame_dict:
                    continue
                next_node_ids = node_frame_dict[t_next]
                next_kdtree, next_positions = create_kdtree(cand_graph, next_node_ids)
                if num_nearest_neighbours is not None:
                    _, matched_indices = next_kdtree.query(
                        x=prev_positions, k=num_nearest_neighbours
                    )
                elif max_edge_distance is not None:
                    matched_indices = prev_kdtree.query_ball_tree(
                        next_kdtree, max_edge_distance
                    )

                for prev_node_id, next_node_indices in zip(
                    prev_node_ids, matched_indices
                ):
                    for next_node_index in next_node_indices:
                        next_node_id = next_node_ids[next_node_index]
                        cand_graph.add_edge(prev_node_id, next_node_id)
