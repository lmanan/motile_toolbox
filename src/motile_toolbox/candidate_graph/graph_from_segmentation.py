import logging
import math
from typing import Any

import networkx as nx
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

from .graph_attributes import EdgeAttr, NodeAttr

logger = logging.getLogger(__name__)


def _get_location(
    node_data: dict[str, Any], position_keys: tuple[str, ...] | list[str]
) -> list[Any]:
    """Convenience function to get the location of a networkx node when each dimension
    is stored in a different attribute.

    Args:
        node_data (dict[str, Any]): Dictionary of attributes of a networkx node.
            Assumes the provided position keys are in the dictionary.
        position_keys (tuple[str, ...] | list[str], optional): Keys to use to get
            location information from node_data (assumes they are present in node_data).
            Defaults to ("z", "y", "x").

    Returns:
        list: _description_
    Raises:
        KeyError if position keys not in node_data
    """
    return [node_data[k] for k in position_keys]


def nodes_from_segmentation(
    segmentation: np.ndarray,
    attributes: tuple[NodeAttr, ...] | list[NodeAttr] = (NodeAttr.SEG_ID,),
    position_keys: tuple[str, ...] | list[str] = ("y", "x"),
    frame_key: str = "t",
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A 3 or 4 dimensional numpy array with integer labels
            (0 is background, all pixels with value 1 belong to one cell, etc.). The
            time dimension is first, followed by two or three position dimensions. If
            the position dims are not (y, x), use `position_keys` to specify the names
            of the dimensions.
        attributes (tuple[str, ...] | list[str] , optional): Set of attributes to
            compute and add to graph nodes. Valid attributes are: "segmentation_id".
            Defaults to  ("segmentation_id",).
        position_keys (tuple[str, ...]| list[str] , optional): What to label the
            position dimensions in the candidate graph. The order of the names
            corresponds to the order of the dimensions in `segmentation`. Defaults to
            ("y", "x").
        frame_key (str, optional): What to label the time dimension in the candidate
            graph. Defaults to 't'.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict = {}

    for t in range(len(segmentation)):
        nodes_in_frame = []
        props = regionprops(segmentation[t])
        for regionprop in props:
            node_id = f"{t}_{regionprop.label}"
            attrs = {
                frame_key: t,
            }
            if NodeAttr.SEG_ID in attributes:
                attrs[NodeAttr.SEG_ID.value] = regionprop.label
            centroid = regionprop.centroid  # [z,] y, x
            for label, value in zip(position_keys, centroid):
                attrs[label] = value
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if nodes_in_frame:
            node_frame_dict[t] = nodes_in_frame
    return cand_graph, node_frame_dict


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    attributes: tuple[EdgeAttr, ...] | list[EdgeAttr] = (EdgeAttr.DISTANCE,),
    position_keys: tuple[str, ...] | list[str] = ("y", "x"),
    frame_key: str = "t",
    node_frame_dict: None | dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        attributes (tuple[str, ...], optional): Set of attributes to compute and add to
            graph.Valid attributes are: "distance". Defaults to ("distance",).
        position_keys (tuple[str, ...], optional): What the position dimensions of nodes
            in the candidate graph are labeled. Defaults to ("y", "x").
        frame_key (str, optional): The label of the time dimension in the candidate
            graph. Defaults to "t".
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    if not node_frame_dict:
        node_frame_dict = {}
        for node, data in cand_graph.nodes(data=True):
            t = data[frame_key]
            if t not in node_frame_dict:
                node_frame_dict[t] = []
            node_frame_dict[t].append(node)
    frames = sorted(node_frame_dict.keys())
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_nodes = node_frame_dict[frame + 1]
        next_locs = [
            _get_location(cand_graph.nodes[n], position_keys=position_keys)
            for n in next_nodes
        ]
        for node in node_frame_dict[frame]:
            loc = _get_location(cand_graph.nodes[node], position_keys=position_keys)
            for next_id, next_loc in zip(next_nodes, next_locs):
                dist = math.dist(next_loc, loc)
                attrs = {}
                if EdgeAttr.DISTANCE in attributes:
                    attrs[EdgeAttr.DISTANCE.value] = dist
                if dist <= max_edge_distance:
                    cand_graph.add_edge(node, next_id, **attrs)


def graph_from_segmentation(
    segmentation: np.ndarray,
    max_edge_distance: float,
    node_attributes: tuple[NodeAttr, ...] | list[NodeAttr] = (NodeAttr.SEG_ID,),
    edge_attributes: tuple[EdgeAttr, ...] | list[EdgeAttr] = (EdgeAttr.DISTANCE,),
    position_keys: tuple[str, ...] | list[str] = ("y", "x"),
    frame_key: str = "t",
):
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance. The specified attributes are computed during construction.
    Node ids are strings with format "{time}_{label id}".

    Args:
        segmentation (np.ndarray): A 3 or 4 dimensional numpy array with integer labels
            (0 is background, all pixels with value 1 belong to one cell, etc.). The
            time dimension is first, followed by two or three position dimensions. If
            the position dims are not (y, x), use `position_keys` to specify the names
            of the dimensions.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_attributes (tuple[str, ...] | list[str], optional): Set of attributes to
            compute and add to nodes in graph. Valid attributes are: "segmentation_id".
            Defaults to ("segmentation_id",).
        edge_attributes (tuple[str, ...] | list[str], optional): Set of attributes to
            compute and add to edges in graph. Valid attributes are: "distance".
            Defaults to ("distance",).
        position_keys (tuple[str, ...], optional): What to label the position dimensions
            in the candidate graph. The order of the names corresponds to the order of
            the dimensions in `segmentation`. Defaults to ("y", "x").
        frame_key (str, optional): What to label the time dimension in the candidate
            graph. Defaults to 't'.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.

    Raises:
        ValueError: if unsupported attribute strings are passed in to the attributes
            arguments, or if the number of position keys provided does not match the
            number of position dimensions.
    """
    if len(position_keys) != segmentation.ndim - 1:
        raise ValueError(
            f"Position labels {position_keys} does not match number of spatial dims "
            f"({segmentation.ndim - 1})"
        )
    # add nodes
    cand_graph, node_frame_dict = nodes_from_segmentation(
        segmentation, node_attributes, position_keys=position_keys, frame_key=frame_key
    )
    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        attributes=edge_attributes,
        position_keys=position_keys,
        node_frame_dict=node_frame_dict,
    )

    logger.info(f"Candidate edges: {cand_graph.number_of_edges()}")
    return cand_graph
