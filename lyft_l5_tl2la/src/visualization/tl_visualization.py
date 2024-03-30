# ------------------------------------------------------------------------
# Extend existing visualization Classes from L5Kit to visualize
# traffic light geometries, traffic light states, traffic light
#  to lane assignments, etc.
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from L5Kit (https://github.com/woven-planet/l5kit)
# Copyright (c) 2022 Woven by Toyota. All Rights Reserved.
#
# Based on files:
#   - l5kit.visualization.visualizer.common
# ------------------------------------------------------------------------


from typing import List, NamedTuple

import numpy as np
from l5kit.visualization.visualizer.common import (AgentVisualization,
                                                   CWVisualization,
                                                   EgoVisualization,
                                                   TrajectoryVisualization)


class TLVisualization(NamedTuple):
    """Hold information about a single traffic light

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param state_color: color of the tl as a type str 
    :param state: state of tl
    :param traffic_light_id: id of traffic light
    :param face_ids: 1D array of traffic light faces
    :param corresponding_lanes: list of corresponding lane ids

    """
    xs: np.ndarray
    ys: np.ndarray
    center_x: np.float
    center_y: np.float
    state_color: str
    state: str
    traffic_light_id: str
    face_ids: np.ndarray
    corresponding_lanes: list


class LaneTLRelationVisualization(NamedTuple):
    """Hold information the relation between a lane and a traffic light

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates

    """
    xs: np.ndarray
    ys: np.ndarray


class LaneVisualization(NamedTuple):
    """Hold information about a single lane

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param color: color of the lane as a string (both hex or text)
    :param id: id of the lane as a string 
    """
    xs: np.ndarray
    ys: np.ndarray
    color: str
    id: str


class NodeVisualization(NamedTuple):
    """Hold information of a node between two segments

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    """
    x: np.float
    y: np.float
    id: str


class JunctionVisualization(NamedTuple):
    """Hold information of a junction

    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    """
    x: np.float
    y: np.float
    id: str


class FrameVisualization(NamedTuple):
    """Hold information about a frame (the state of a scene at a given time)

    :param ego: a single ego annotation
    :param agents: a list of agents
    :param lanes: a list of lanes
    :param crosswalks: a list of crosswalks
    :param trajectories: a list of trajectories
    :param trafficLights: a list of traffic lights
    :param lane_tl_relations: a list of lane traffic light relations
    """
    ego: EgoVisualization
    agents: List[AgentVisualization]
    lanes: List[LaneVisualization]
    crosswalks: List[CWVisualization]
    trajectories: List[TrajectoryVisualization]
    trafficLights: List[TLVisualization]
    lane_tl_relations: List[LaneTLRelationVisualization]
    nodes: List[NodeVisualization]


class StaticVisualization(NamedTuple):
    """Hold information about a frame (the state of a scene at a given time)

    :param lanes: a list of lanes
    :param crosswalks: a list of crosswalks
    :param trafficLights: a list of traffic lights
    :param lane_tl_relations: a list of lane traffic light relations
    :param tl_junctions: a list of junctions that get controlled by traffic lights
    """
    lanes: List[LaneVisualization]
    crosswalks: List[CWVisualization]
    trafficLights: List[TLVisualization]
    lane_tl_relations: List[LaneTLRelationVisualization]
    predicted_lane_tl_relations: List[LaneTLRelationVisualization]
    gt_lane_tl_relations: List[LaneTLRelationVisualization]
    nodes: List[NodeVisualization]
    tl_junctions: List[JunctionVisualization]
