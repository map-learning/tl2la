# ------------------------------------------------------------------------
# Utility functions to handle Aggregate, Filter, Fetch Data from 
# Lyft Lvl 5 Dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from l5kit.data import ChunkedDataset
from l5kit.data.zarr_dataset import AGENT_DTYPE
from l5kit.geometry import rotation33_as_yaw
from l5kit.rasterization.box_rasterizer import (EGO_EXTENT_HEIGHT,
                                                EGO_EXTENT_LENGTH,
                                                EGO_EXTENT_WIDTH,
                                                get_box_world_coords)
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from loguru import logger
from shapely.geometry import Polygon
from src.common.definitions import (SAMPLE_FREQUENCY, STATUS_CHANGE)
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import calculate_distance
from tqdm import tqdm



def get_ego_as_agent(ego_pos: np.ndarray, ego_rotation: np.ndarray) -> np.ndarray:
    """Transfroms ego numpy array into agents numpy array.

    Args:
        ego_pos (np.ndarray): ego position
        ego_rotation (np.ndarray): ego rotation

    Returns:
        np.ndarray: of type agent
    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = ego_pos[:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(ego_rotation)
    ego_agent[0]["extent"] = np.asarray(
        (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


def get_ego_rect(ego_pos: np.ndarray, ego_rotation: np.ndarray) -> Polygon:
    """Returns the Polygon of the ego AV

    Args:
        ego_pos (np.ndarray): ego position
        ego_rotation (np.ndarray): ego rotation

    Returns:
        Polygon: bounding box
    """
    ego_agent = get_ego_as_agent(ego_pos, ego_rotation)
    box_world_coords = get_box_world_coords(ego_agent)
    points = box_world_coords[0, :, :]
    return Polygon((points[0], points[1], points[2], points[3]))


def get_agent_rect(agent: np.ndarray) -> Polygon:
    """Returns the Polygon of the ego an agent rect
    Args:
        agent (np.ndarray): agent array

    Returns:
        Polygon: agent bounding box
    """
    box_world_coords = get_box_world_coords(np.array([agent]))
    points = box_world_coords[0, :, :]
    return Polygon((points[0], points[1], points[2], points[3]))


def get_lane_poly(lane_id: str, mapAPI: ExtendedMapAPI) -> Polygon:
    """Generates a shapely Polygon out of the lane coordinates of a given lane

    Args:
        lane_id (str): element id of lane 
        mapAPI (ExtendedMapAPI): map API

    Returns:
        Polygon: Lane Polygon
    """
    lane_coords = mapAPI.get_lane_coords(lane_id)

    lane_poly = np.concatenate(
        (lane_coords["xyz_left"][:, :2], lane_coords["xyz_right"][::-1, :2]))
    return Polygon(lane_poly)


def visualize_mapping(ego_rect: Polygon, lane_poly: Polygon):
    """Visualise the mapping of a eg vehicle and a lane

    Args:
        point (Point): ego xy-pos
        poly (Polygon): lane shape
    """

    plt.plot(*lane_poly.exterior.xy)
    plt.plot(*ego_rect.exterior.xy)
    plt.title("Check if ego on lane")
    plt.show()


def visualize_lane(lane_id: str, mapAPI: ExtendedMapAPI):
    """Visualise a lane

    Args:
        lane_id (str): id of lane element
    """
    lane_poly: Polygon = get_lane_poly(lane_id, mapAPI)
    plt.plot(*lane_poly.exterior.xy)
    plt.title(f"Lane id: {lane_id}")
    plt.show()

def visualize_tl_status(tl_status: pd.DataFrame, scene_idx: int, specific_ids: Optional[list] = None, path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))

    tl_mapping_scene_df = tl_status[tl_status["scene_idx"]==scene_idx]
    if specific_ids is None:
        traffic_light_ids = tl_mapping_scene_df["tl_id"].unique()
    else:
        traffic_light_ids = specific_ids

    for tl_id in traffic_light_ids:
        current_tl_mapping_df = tl_mapping_scene_df[tl_mapping_scene_df["tl_id"] == tl_id]
        sns.lineplot(x="frame_idx", y="tl_status", legend=True,
                    ax=ax, data=current_tl_mapping_df, label=tl_id)

    plt.title(f"Tl State in Scene: {scene_idx}")
    plt.xlabel("Frame")
    plt.ylabel("State")
    plt.legend(title="Tl id")
    plt.grid()
    plt.ylim((-0.5,4.5))
    if path: plt.savefig(path, format="png", dpi=1000)
    plt.show()

def key_with_max_val(d: dict):
    """Returns the key with the max value
   """
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def map_agent_to_lanes(ego_or_agent: np.ndarray, mapAPI: ExtendedMapAPI, scene_idx: int, frame_idx: int, is_ego: bool) -> List:
    """Maps given position of agents to a lane. Currently only implemented for ego. TODO: implement for other agents with thei bounding box

    Args:
        agent_pos (np.ndarray): position of agent
        mapAPI (ExtendedMapAPI): API for semantic map interaction

    Returns:
        str: lane id of current AV position
    """
    SEARCH_RADIUS = 2

    if is_ego:
        agent_pos = ego_or_agent["ego_translation"]
        agent_rotation = ego_or_agent["ego_rotation"]
        agent_rect = get_ego_rect(agent_pos, agent_rotation)
    else:
        # transoform XYZ format
        agent_pos = np.append(ego_or_agent["centroid"], 0)
        agent_rect = get_agent_rect(ego_or_agent)

    lane_ids: list = mapAPI.bounds_info["lanes"]["ids"]
    lane_indices = indices_in_bounds(
        agent_pos[:2], mapAPI.bounds_info["lanes"]["bounds"], SEARCH_RADIUS)
    on_lanes = []
    intersection_area = dict()  # key: lane_id, value: intersection areas
    for lane_idx in lane_indices:
        lane_id = lane_ids[lane_idx]
        lane_coords = mapAPI.get_lane_coords(lane_id)

        lane_poly_coords = np.concatenate(
            (lane_coords["xyz_left"][:, :2], lane_coords["xyz_right"][::-1, :2]))
        lane_poly = Polygon(lane_poly_coords)

        if agent_rect.centroid.within(lane_poly):
            on_lanes.append(lane_id)

        elif agent_rect.intersects(lane_poly):
            try:
                intersection_area[lane_id] = agent_rect.intersection(
                    lane_poly).area
            except ValueError as e:
                logger.info(f"Intersection not possble: {e}")

    if len(on_lanes):
        return on_lanes
    if len(on_lanes) == 0 and intersection_area:  # dict not empty and intersections found
        return list(intersection_area.keys())

    logger.debug(
        f"Mapping for Agent Pos: {agent_pos[0:2]} was not possible in Scene: {scene_idx}, Frame: {frame_idx}!")
    return []


def filter_tl_scenes(zarr_dataset: ChunkedDataset) -> list:
    """Filters out the traffic light scenes and returns the scene idx

    Args:
        zarr_dataset (ChunkedDataset): given dataset

    Returns:
        list: indices of traffic light scenes
    """
    
    scene_idxs_with_tfls = []
    for scene_idx in tqdm(range(len(zarr_dataset.scenes)), desc="Iterate through scenes to get scene idxs with tfls"):
        current_scene = zarr_dataset.get_scene_dataset(scene_index=scene_idx)

        # add scenes with tfls
        if len(current_scene.tl_faces) > 0:
            scene_idxs_with_tfls.append(scene_idx)

    return scene_idxs_with_tfls


def filter_tl_status_change(tl_mapping: pd.DataFrame, status_change: str) -> pd.DataFrame:
    """Filters out the changed rows of a specific traffic light status change

    Args:
        tl_mapping (pd.DataFrame): dataframe with traffic light mapping
        status_change (str): the relevant status change

    Returns:
        pd.DataFrame: filtered dataframe with scen and frame index of the status change
    """

    return tl_mapping[tl_mapping["tl_status"].diff() == STATUS_CHANGE[status_change]]


def detect_active_status_change(tl_mapping: pd.DataFrame) -> pd.DataFrame:
    """Returns the index of a frame when the traffic light status changes 

    Args:
        tl_mapping (pd.DataFrame): _description_

    Returns:
        dict: tl_id frame idx of change
    """
    tl_mapping = tl_mapping.sort_values(["tl_id", "scene_idx", "frame_idx"])
    unique_tl = tl_mapping["tl_id"].unique()
    status_change = []

    for tl_id in unique_tl:
        current_tl_mapping = tl_mapping[tl_mapping["tl_id"] == tl_id]
        red_to_green_change = current_tl_mapping[current_tl_mapping["tl_status"].diff(
        ) == 1]
        green_to_red_change = current_tl_mapping[current_tl_mapping["tl_status"].diff(
        ) == -1]

        for idx, row in red_to_green_change.iterrows():
            status_change.append([row["scene_idx"], row["frame_idx"],
                                 row["tl_id"], STATUS_CHANGE["RED_TO_GREEN"], "RED_TO_GREEN"])

        for idx, row in green_to_red_change.iterrows():
            status_change.append([row["scene_idx"], row["frame_idx"],
                                 row["tl_id"],  STATUS_CHANGE["GREEN_TO_RED"], "GREEN_TO_RED"])

    return pd.DataFrame(
        status_change,
        columns=["scene_idx", "frame_idx", "tl_id",
                 "status_change", "status_change_str"]
    )


def calculate_ego_velocity(xyz: pd.DataFrame, time: pd.Series) -> pd.Series:
    """Calculate velocity for given xyz position between timestamps

    Args:
        xyz (pd.DataFrame): xyz position

    Returns:
        pd.Series: velocity vector, first element Nan!
    """

    X1 = xyz.to_numpy()
    X0 = xyz.shift(1).to_numpy()
    X_d = (X1 - X0)
    X_d = X_d[1:, ]
    dist = np.sum(X_d**2, axis=1)**0.5
    dist = np.insert(dist, 0, np.NaN)
    return dist / (time-time.shift(1))


def calculate_agent_velocity(agent_velocity_xy: np.ndarray) -> np.ndarray:
    """Calculates the velocity of N agent

    Args:
        agent_velocity_xy (np.ndarray): velocity vector of N agent in x and y direction

    Returns:
        float: scalar velocity of agents
    """
    return np.sum(agent_velocity_xy**2, axis=1)**0.5


def calcualte_agent_acceleration(agent_velocity_frame: pd.DataFrame) -> pd.Series:
    """Calcualte the acceleration of the agents

    Args:
        agent_velocity_frame (pd.DataFrame): velcity frame with time information of traffic agents

    Returns:
        pd.Series: acceleration of traffic agents
    """
    unique_agent_ids = agent_velocity_frame["agent_id"].unique()
    acceleration_list = []
    for agent_id in unique_agent_ids:
        current_agent_velocity_df = agent_velocity_frame[agent_velocity_frame["agent_id"] == agent_id]
        accel_series = (current_agent_velocity_df["v"]-current_agent_velocity_df["v"].shift(1)) / \
            ((current_agent_velocity_df["time"] -
             current_agent_velocity_df["time"].shift(1)))  # .rolling(25, center=True).mean() # & apply moving average
        acceleration_list.append(accel_series)
    return pd.concat(acceleration_list)


def get_ego_movement_frame(ego_translation: np.ndarray, scene_idx: int, num_frames: int) -> pd.DataFrame:
    """Generate Dataframe with velocity and time information of ego vehicle

    Args:
        ego_translation (np.ndarray): array of xyz position of ego vehilce
        scene_idx (int): index of current scene
        num_frames (int): number of frames in scene

    Returns:
        (pd.Dataframe): velocity dataframe of ego vehicle
    """
    ego_velocity_df = pd.DataFrame(ego_translation, columns=["x", "y", "z"])
    time = np.arange(
        0, num_frames*SAMPLE_FREQUENCY, SAMPLE_FREQUENCY)
    ego_velocity_df["time"] = time[:num_frames]
    ego_velocity_df.insert(0, "scene_idx", [scene_idx]*num_frames)
    ego_velocity_df.insert(1, "frame_idx", np.arange(0, num_frames))
    ego_velocity_df.insert(2, "agent_id", 0)
    ego_velocity_df["v"] = calculate_ego_velocity(ego_velocity_df[["x", "y", "z"]],
                                                  ego_velocity_df["time"]
                                                  )
    ego_velocity_df["a"] = (ego_velocity_df["v"]-ego_velocity_df["v"].shift(1)) / \
        ((ego_velocity_df["time"]-ego_velocity_df["time"].shift(1))
         )  # .rolling(25, center=True).mean()  # & apply moving average
    ego_velocity_df["centroid"] = ego_velocity_df.apply(
        lambda row: np.array([row["x"], row["y"]]), axis=1)
    ego_velocity_df = ego_velocity_df.drop(["x", "y", "z"], axis=1)
    return ego_velocity_df


def get_agents_velocity_frame(agent_frames: np.ndarray, scene_idx: int) -> pd.DataFrame:
    """Generate Dataframe with velocity and time information of surrounding traffic agents

    Args:
        agent_frames (np.ndarray): agent frame data with velocity information
        scene_idx (int): index of current scene

    Returns:
        (pd.Dataframe): velocity dataframe of agent vehicles
    """
    agent_velocity_df = pd.DataFrame(
        columns=["scene_idx", "frame_idx", "agent_id", "time", "v", "centroid"])
    AGENT_FRAMES = len(agent_frames)

    for agent_frame_idx in range(AGENT_FRAMES):
        agents = agent_frames[agent_frame_idx]

        velocity = calculate_agent_velocity(agents["velocity"])

        time = agent_frame_idx*SAMPLE_FREQUENCY
        agent_velocity = np.column_stack(([scene_idx]*len(agents), [agent_frame_idx]*len(agents),
                                          agents["track_id"].tolist(), [time]*len(agents), velocity.tolist(), [0]*len(agents)))

        helper_df = pd.DataFrame(
            agent_velocity, columns=["scene_idx", "frame_idx", "agent_id", "time", "v", "centroid"])
        helper_df["centroid"] = pd.Series(agents["centroid"].tolist())

        agent_velocity_df = agent_velocity_df.append(
            helper_df, ignore_index=True)

        agent_velocity_df = agent_velocity_df.astype(
            {"scene_idx": "int", "frame_idx": "int", "agent_id": "int"})
    return agent_velocity_df


def get_movement_frame(scene_idx: int, frames: np.ndarray, agent_frames: np.ndarray) -> pd.DataFrame:
    """Get for a specific scene the a agent and ego movement. This includes, velocity, acceleration and centroid
    for each frame in the scene.

    Args:
        scene_idx (int): index of scene
        frames (np.ndarray): frames
        agent_frames (np.ndarray): agent frames

    Returns:
        pd.DataFrame: movement dataframe
    """

    ego_movement_df = get_ego_movement_frame(frames["ego_translation"],
                                             scene_idx,
                                             len(frames)
                                             )
    agent_movement_df = get_agents_velocity_frame(agent_frames, scene_idx)
    agent_movement_df["a"] = calcualte_agent_acceleration(agent_movement_df)

    movement_df = pd.concat(
        [ego_movement_df, agent_movement_df], ignore_index=True)
    return movement_df


def merge_agent_lane_mapping_and_branch_frame(agent_lane_mapping_df: pd.DataFrame, extended_branch_df: pd.DataFrame) -> pd.DataFrame:
    """Merge agent lane mapping dataframe with extended branch dataframe

    Args:
        agent_lane_mapping_df (pd.DataFrame): agent to lane mapping dataframe
        extended_branch_df (pd.DataFrame): branch dataframe with lane sequences

    Returns:
        pd.DataFrame: merged dataframe
    """
    agent_lane_mapping_df = agent_lane_mapping_df.explode('lane_ids')
    agent_lane_mapping_df.rename(columns={'lane_ids': 'lane_id'}, inplace=True)

    agent_mapping_branch_df = agent_lane_mapping_df.join(
        extended_branch_df.set_index('lane_id'), on=['lane_id'])
    agent_mapping_branch_df = agent_mapping_branch_df.dropna()
    agent_mapping_branch_df = agent_mapping_branch_df.drop_duplicates()
    return agent_mapping_branch_df


def agent_on_branch(current_agent_mapping: pd.DataFrame, branch_id: str) -> bool:
    """Check if agent on branch with given id

    Args:
        branch_id (str): branch id
        current_agent_mapping (pd.Dataframe): mapping of current agent to lane

    Returns:
        bool: checks if on branch
    """
    return len(current_agent_mapping[current_agent_mapping["branch_id"] == branch_id]) > 0


def on_lane_sequence(current_agent_mapping: pd.DataFrame) -> bool:
    """Check if the agent drives on a lane that is part of a lane sequence of a branch.
    -1 indicates that the lane segment is not part of a sequence!

    Args:
        current_agent_mapping (pd.Dataframe): mapping of current agent to lane

    Returns:
        bool: true if it has common lane idx
    """
    return len(current_agent_mapping[current_agent_mapping["lane_sequence_id"] != -1.0]) > 0


def above_given_track_length(agent_vel: pd.DataFrame, threshold: int) -> bool:
    """Check if agents track length is above a given threshold

    Args:
        agent_vel (pd.DataFrame): _description_
        threshold (int): minimum given track length

    Returns:
        bool: is track length over threshold?
    """
    start_time = agent_vel["time"].iloc[0]
    end_time = agent_vel["time"].iloc[-1]
    return (end_time-start_time) >= threshold


def above_given_speed(agent_vel: pd.DataFrame, threshold: int) -> bool:
    """Check if agents speed is above a given threshold

    Args:
        agent_vel (pd.DataFrame): _description_
        threshold (int): reached speed

    Returns:
        bool: is track length over threshold?
    """
    return (agent_vel["v"] >= threshold).any()


def get_tl_distance(mapAPI: ExtendedMapAPI, lane_head_id: str, centroid: np.ndarray) -> np.float:
    """Calculate for each agent the distance towards the stopping line depending on its current lane sequence

    Args:
        mapAPI (ExtendedMapAPI): map Api
        lane_head_id (str): head lane id of current lane sequence 
        centroid (np.ndarray): agent's current centroid

    Returns:
        (np.float, None): distance to stopping line
    """
    if not isinstance(lane_head_id, str):
        return np.NaN
    return calculate_distance(mapAPI.get_lane_stopping_point(lane_head_id), centroid)


def calculate_tl_distance(mapAPI: ExtendedMapAPI, motion_df: pd.DataFrame) -> pd.DataFrame:
    motion_df["dist"] = np.NaN
    motion_df["dist"] = motion_df.apply(lambda x: get_tl_distance(
        mapAPI, x.get("sequence_head_lane_id"), x.get("centroid")), axis=1)
    return motion_df



def calculate_lead_vehicle(motion_df: pd.DataFrame) -> pd.DataFrame:
    """Checks if vehicle is the first vehicle infront of the stopping line.

    Args:
        motion_df (pd.DataFrame): dataframe with motion data and distance information towards lead vehicle

    Returns:
        pd.DataFrame: _description_
    """
    # calculate lead indices
    lead_indices = motion_df.groupby(["scene_idx", "frame_idx", "sequence_head_lane_id"])["dist"].idxmin()
    motion_df["is_lead"] = False
    motion_df.loc[lead_indices.values.tolist(), "is_lead"] = True
    return motion_df


def calculate_traffic_density(motion_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate traffic density of a lane sequence

    Args:
        motion_df (pd.DataFrame): motion dataframe

    Returns:
        pd.DataFrame: motion dataframe with appended density
    """
    motion_df["density"] = np.NaN
    density = motion_df.groupby(["scene_idx", "frame_idx", "sequence_head_lane_id"])['agent_id'].transform('count')
    motion_df["density"] = density
    return motion_df.astype({"density":"int"})


def calculate_elapsed_time_in_tl_state(scene_idx: int, frame_idx: int, status_change_df: pd.DataFrame, tl_id: str) -> float:
    """Calcualtes the time elapsed after a state change of a traffic light

    Args:
        frame_idx (int): index of current frame
        status_change_df (pd.DataFrame): tl state change information 

    Returns:
        float: elapsed time
    """
    change_frame_indices = status_change_df[(status_change_df["scene_idx"] == scene_idx) & (
        status_change_df["tl_id"] == tl_id)]["frame_idx"].values
    change_frame_distance = frame_idx-change_frame_indices
    positive_change_frame_distance = change_frame_distance[change_frame_distance >= 0]

    # minimum elapsed time in traffic light state since the beginning of the scene if no state change was seen
    if len(positive_change_frame_distance) == 0:
        return frame_idx * SAMPLE_FREQUENCY
    # otherwise calculate time between last state change
    idx = positive_change_frame_distance.argmin()
    distance = positive_change_frame_distance[idx]
    return distance * SAMPLE_FREQUENCY


def generate_tl_lane_relation_lookup(ground_trouth: pd.DataFrame) -> dict:
    """Generate a fast lookup table for the traffic light to lane relation using the ground trouth relation

    Args:
        ground_trouth (pd.DataFrame): ground trouth relation of traffic lights and lanes

    Returns:
        dict: lookup
    """
    ground_trouth_dict = dict()
    for _ , row in ground_trouth.iterrows():
        hash = row["tl_id"] + row["lane_id"]
        ground_trouth_dict[hash] = row["relation"]
    return ground_trouth_dict


def get_relation_by_lookuptable(tl_id: str, lane_id: str, relation_lookup: dict) -> int:
    """Get the traffic light to lane relation (1 or 0) of a specific traffic light and a lane by using a lookup table.

    Args:
        tl_id (str): traffic light id
        lane_id (str): lane id
        relation_lookup (dict): lookup

    Returns:
        int: relation of traffic light and lane
    """
    hash = tl_id+lane_id
    if hash in relation_lookup.keys():
        return relation_lookup[hash]
    return -2


