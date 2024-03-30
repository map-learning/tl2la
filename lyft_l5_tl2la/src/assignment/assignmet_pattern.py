# ------------------------------------------------------------------------
# Implementation of Pattern-based contribution Method
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
from src.common.definitions import (AGENT_ACCELERATION_STATES, AGENT_PATTERN,
                                    AGENT_VELOCITY_STATES, DATASET_TYPE,
                                    SAMPLE_FREQUENCY, SCENE_TL_STATUS,
                                    THRESHOLDS, TL_STATUS, TURN)
from src.common.helper import get_pickled_dataset_index
from src.common.utils import detect_active_status_change
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.dataset_operations.junction_dataset import JunctionDataset


def get_elapsed_time_in_tl_state(frame_idx: int, tl_mapping: pd.DataFrame) -> float:
    """Calcualtes the time elapsed after a state change of a traffic light

    Args:
        frame_idx (int): index of current frame
        tl_mapping (pd.DataFrame): tl mapping with state information for whole frame

    Returns:
        float: elapsed time
    """
    status_change_df = detect_active_status_change(tl_mapping)
    status_change_bevore_current_frame = status_change_df[status_change_df["frame_idx"] < frame_idx]
    if len(status_change_bevore_current_frame) == 0:
        return frame_idx * SAMPLE_FREQUENCY
    last_status_change = status_change_bevore_current_frame.iloc[-1]
    return (frame_idx - last_status_change["frame_idx"]) * SAMPLE_FREQUENCY


def calculate_contribution(pattern: int, tl_status: int, tl_status_time: np.float, is_lead: bool, dist: float, turn_type: int, v: float) -> float:
    """Calculate contribution value for detectet agent pattern and tl state

    Args:
        pattern (int): detected agent pattern
        tl_status (int): state of traffic light
        tl_status_time (np.float): time elapsed in tl state

    Raises:
        Exception: if given pattern is not defined

    Returns:
        int: contribution value
    """
    # TODO: may be useful to weight the contribution values by distance
    
    if pattern == AGENT_PATTERN.STATONARY.value:
        if tl_status == TL_STATUS.RED.value and dist < THRESHOLDS.T_STOP_ZONE.value:
            return 2
        elif is_lead and tl_status_time > THRESHOLDS.T_GREEN_REACTION.value and tl_status == TL_STATUS.GREEN.value:
            return -3
        elif is_lead and tl_status == TL_STATUS.GREEN.value:
            return -1
        return 0  # no contribution
    elif pattern == AGENT_PATTERN.MOVING.value:
        if tl_status == TL_STATUS.RED.value and tl_status_time > THRESHOLDS.T_RED_REACTION.value and dist < THRESHOLDS.T_STOP_ZONE.value:
            return -3
        elif tl_status == TL_STATUS.RED.value and dist < THRESHOLDS.T_SD_ZONE.value:
            return -1
        elif tl_status == TL_STATUS.GREEN.value and dist < THRESHOLDS.T_SD_ZONE.value:
            return 3
        elif tl_status == TL_STATUS.GREEN.value and dist < THRESHOLDS.T_STOP_ZONE.value:
            return 5
        return 0
    elif pattern == AGENT_PATTERN.ACCELERATION.value:
        # from moving
        if v in AGENT_VELOCITY_STATES.V_MOVING.value:
            if tl_status == TL_STATUS.RED.value and dist < THRESHOLDS.T_STOP_ZONE.value and tl_status_time > THRESHOLDS.T_RED_REACTION.value:
                return -3
            elif tl_status == TL_STATUS.RED.value and dist < THRESHOLDS.T_SD_ZONE.value:
                return -1
            elif tl_status == TL_STATUS.GREEN.value and dist < THRESHOLDS.T_STOP_ZONE.value:
                return 1
            return 0
        else:  # from stationary
            if tl_status == TL_STATUS.RED.value and dist < THRESHOLDS.T_STOP_ZONE.value and turn_type != TURN.RIGHT.value:
                return -2
            elif tl_status == TL_STATUS.GREEN.value and dist < THRESHOLDS.T_SD_ZONE.value and tl_status_time > THRESHOLDS.T_GREEN_REACTION.value:
                return 3
            return 0
    elif pattern == AGENT_PATTERN.DECELERATION.value:
        if tl_status == TL_STATUS.RED.value and dist > THRESHOLDS.T_STOP_ZONE.value and dist < THRESHOLDS.T_SD_ZONE.value:
            return 2
        elif tl_status == TL_STATUS.GREEN.value and is_lead and dist < THRESHOLDS.T_STOP_ZONE.value and turn_type == TURN.LEFT.value:
            return -1
        elif tl_status == TL_STATUS.GREEN.value and is_lead and dist < THRESHOLDS.T_STOP_ZONE.value and turn_type == TURN.NORMAL.value:
            return -2
        return 0
    elif pattern == AGENT_PATTERN.UNDEFINED.value:
        return 0
    raise Exception(f"Please check pattern: {pattern}")


def detect_pattern(dist: np.float, velocity: np.float, acceleration: np.float) -> AGENT_PATTERN:
    """Determine agent pattern based on velocity, acceleration and distance of agent to stopping line

    Args:
        dist (np.float): distance between agent and stopping line
        velocity (np.float): velocity of agent
        acceleration (np.float): acceleration of agent

    Returns:
        Optional[int]: the detected pattern
    """
    if dist > THRESHOLDS.T_DIST.value:
        return AGENT_PATTERN.UNDEFINED
    if np.isnan(velocity) or np.isnan(acceleration):
        return AGENT_PATTERN.UNDEFINED

    # Check for pattern
    if velocity in AGENT_VELOCITY_STATES.V_STANDSTIL.value:
        if acceleration in AGENT_ACCELERATION_STATES.ACCELERATING.value:
            return AGENT_PATTERN.ACCELERATION
        elif acceleration in AGENT_ACCELERATION_STATES.DECELERATION.value:
            return AGENT_PATTERN.STATONARY
        elif acceleration in AGENT_ACCELERATION_STATES.NO_ACCELERATION.value:
            return AGENT_PATTERN.STATONARY
        return AGENT_PATTERN.UNDEFINED
    elif velocity in AGENT_VELOCITY_STATES.V_MOVING.value:
        if acceleration in AGENT_ACCELERATION_STATES.ACCELERATING.value:
            return AGENT_PATTERN.MOVING
        if acceleration in AGENT_ACCELERATION_STATES.DECELERATION.value:
            return AGENT_PATTERN.DECELERATION
        elif acceleration in AGENT_ACCELERATION_STATES.NO_ACCELERATION.value:
            return AGENT_PATTERN.MOVING
        return AGENT_PATTERN.UNDEFINED
    return AGENT_PATTERN.UNDEFINED  # other states are undefined -> no pattern detected!


def fetch_contribution(dist: float, velocity: float, acceleration: float, tl_state: int, time_in_tl_state: float,  is_lead: bool, turn_type: int):
    """After grouping by: scene_idx, frame_idx, sequence_head_lane_id, tl_id, agent_id

    Args:
        combined_features (pd.DataFrame): complete dataset with all features
        mapAPI (ExtendedMapAPI): map API Object to interact with HD map
    """
    agent_pattern = detect_pattern(dist, velocity, acceleration)
    return calculate_contribution(agent_pattern.value, tl_state, time_in_tl_state, is_lead, dist, turn_type, velocity)


def heuristic_data_preprocessing(combined_features: pd.DataFrame, map_api: ExtendedMapAPI, scene_tl_state: pd.DataFrame, extended_branch_df: pd.DataFrame, only_tl_lane_pairs: Optional[pd.DataFrame] = None, only_ego: bool = False) -> pd.DataFrame:
    """Data preprocessing for heuristic approach

    Args:
        combined_features (pd.DataFrame): uncleaned dataset
        map_api (ExtendedMapAPI): map API
        scene_tl_state (pd.DataFrame): dataset containing interpreted traffic light scene state
        extended_branch_df (pd.DataFrame): dataset containing junction, branches and lanes
        only_ego (bool): consider only ego or all agents

    Returns:
        pd.Dataframe: cleaned and combined dataset
    """
    if only_ego: 
        combined_features = combined_features.query("agent_id == 0")
        
    combined_features = combined_features.query("v >= 0.5")
    combined_features = combined_features.query(f"tl_status != {TL_STATUS.UNKNOWN.value}")
    
    # filter out traffic lights with invalid state
    combined_features = combined_features.merge(scene_tl_state, on=["scene_idx", "tl_id"], how="left")
    combined_features = combined_features[~combined_features.isnull().any(axis=1)]
    combined_features = combined_features[combined_features["interpreted_state"]!=SCENE_TL_STATUS.UNDEFINED.value]
    
    #filter out roundtrip bug
    combined_features = combined_features.merge(extended_branch_df, on=["junction_id", "branch_id", "lane_id", "sequence_head_lane_id"], how="left")
    combined_features = combined_features.dropna(subset="num_sequence_head_lane_ids") 
    
    # filter out specific lanes
    ignore_lane_ids = ["qgZI", # roundtrip
                       "GRMC", "nQMC", "IRMC", # overlapping lanes infront of junction: E0Us
                       "yISO", "TISO", "RJSO", "/w/Q", "9x/Q", "ex/Q" # overlapping lanes infront of junction: ctKJ
                        "OLcZ", "TbHk", "ybHk", "v7zs", "o7Zz" # overlapping lanes west arm of junction y4ss
                       ]
    combined_features = combined_features[~combined_features["lane_id"].isin(ignore_lane_ids)]
    
    # only consider specific tl lane pairs
    if only_tl_lane_pairs is not None:
        only_tl_lane_pairs = only_tl_lane_pairs.rename(columns={"lane_id": "sequence_head_lane_id"})
        combined_features = combined_features.merge(only_tl_lane_pairs, on=["sequence_head_lane_id", "tl_id"])

    # Add lane type
    combined_features["turn_type"] = combined_features["lane_id"].apply(map_api.get_turn_type)
    
    combined_features = combined_features.astype({"state_known_start_frame": 'int', "state_known_end_frame": 'int', "state_change_frame": 'int', "interpreted_state": "int"})
    return combined_features

def apply_frame_contribution(combined_features: pd.DataFrame) -> pd.DataFrame:
    """Calculate contribution for every traffic light - agent - lane combination frame-wise.

    Args:
        combined_features (pd.DataFrame): dataset with all features

    Returns:
        pd.DataFrame: dataset with contribution for TL2LA betwenn tl and lane pair
    """
    group_labels = ["scene_idx", "frame_idx", "sequence_head_lane_id", "tl_id", "agent_id"]
    sample_scene_grouped = combined_features.groupby(by=group_labels, as_index=True)
    
    contributions = sample_scene_grouped[["dist", "v", "a", "tl_status", "time_in_tl_state", "is_lead", "turn_type"]].apply(lambda group: fetch_contribution(*group.values[0]))
    contributions_df = contributions.reset_index().rename(columns={0: "contribution"})
    combined_features = combined_features.merge(contributions_df, on=group_labels)
    combined_features = combined_features.dropna(subset="contribution")
    return combined_features

def generalize_over_frames(frame_contribution: pd.DataFrame) -> pd.DataFrame:
    """Generalization of contribution over frames -> predict TL2LA per scene by aggregating the contribution scene-wise

    Args:
        frame_contribution (pd.DataFrame): dataset with frame contribution

    Returns:
        pd.DataFrame:  dataset with scene prediction
    """
    contribution_scenes = frame_contribution.groupby(["scene_idx", "sequence_head_lane_id", "tl_id"], as_index=False).aggregate({"contribution": "sum"}).astype({"contribution": "int"})
    contribution_scenes["scene_relation"] = contribution_scenes["contribution"].apply(lambda x: 1 if x > 0 else 0)
    return contribution_scenes

def generalize_over_scenes_by_relation(scene_contribution: pd.DataFrame, significance: float = 0.5) -> pd.DataFrame:
    """Generalization of predictions over scenes by average relation -> predict TL2LA by aggregating the prediction of single scenes

    Args:
        scene_contribution (pd.DataFrame): dataset with scene prediction

    Returns:
        pd.DataFrame:  dataset with TL2LA based on the provided scenes
    """
    contribution_scenes_agg = scene_contribution.groupby(by=["sequence_head_lane_id", "tl_id"], as_index=False).aggregate({"scene_relation": "sum"})
    average_scene_relations = contribution_scenes_agg.groupby(by=["sequence_head_lane_id"], as_index=False).aggregate({"scene_relation": "median"}).rename(columns={"scene_relation": "avg_scene_relation"})
    contribution_scenes_agg = contribution_scenes_agg.merge(average_scene_relations, on=["sequence_head_lane_id"])
    contribution_scenes_agg["predicted_relation"] = contribution_scenes_agg.apply(lambda row: 1 if (row["scene_relation"] > significance * row["avg_scene_relation"]) else 0, axis=1)
    return contribution_scenes_agg

def generalize_over_scenes_by_class_majority(scene_contribution: pd.DataFrame, significance: float = 0.5) -> pd.DataFrame:
    """Generalization of predictions over scenes by class majority -> predict TL2LA by aggregating the prediction of single scenes

    Args:
        scene_contribution (pd.DataFrame): dataset with scene prediction

    Returns:
        pd.DataFrame:  dataset with TL2LA based on the provided scenes
    """
    pos_assignment_counts = scene_contribution.groupby(by=["sequence_head_lane_id", "tl_id"], as_index=False).aggregate({"scene_relation": "sum"}).rename(columns={"scene_relation": "pos_assignment_counts"})
    all_assignments_counts = scene_contribution.groupby(by=["sequence_head_lane_id", "tl_id"], as_index=False).aggregate({"scene_relation": "count"}).rename(columns={"scene_relation": "all_assignments_counts"})
    contribution_scenes_agg = all_assignments_counts.merge(pos_assignment_counts, on=["sequence_head_lane_id", "tl_id"])
    contribution_scenes_agg["predicted_relation"] = contribution_scenes_agg["pos_assignment_counts"] >= significance * contribution_scenes_agg["all_assignments_counts"]
    contribution_scenes_agg["predicted_relation"] = contribution_scenes_agg["predicted_relation"].map({True: 1, False: 0})
    return contribution_scenes_agg

def heristic_prediction_for_signle_scene(scene_idx: int, cfg: dict, junctionDataset: JunctionDataset, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Apply the heuristic approach on given scenes

    Args:
        scene_idx (int): Scene to predict TL2LA
        cfg (dict): config 

    Returns:
        pd.DataFrame: prediction based on the scene
    """
    
    features = junctionDataset.get_combined_features( get_pickled_dataset_index(scene_idx, cfg)).query("scene_idx == @scene_idx") 
    features["turn_type"] = features["lane_id"].apply(mapAPI.get_turn_type)

    # calculation contribution
    frame_contribution = apply_frame_contribution(features)

    scene_prediction = generalize_over_frames(frame_contribution)
    prediction = generalize_over_scenes_by_class_majority(scene_prediction)
    
    # Append intersection information
    prediction = prediction.rename(columns={"sequence_head_lane_id": "lane_id"})
    prediction = prediction.merge(junctionDataset.get_branch_dataset(DATASET_TYPE.MINIMAL), on ="lane_id", how="left")
    return prediction


def heristic_prediction_for_scenes(scene_indices: List[int], cfg: dict, junctionDataset: JunctionDataset, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Apply the heuristic approach on given scenes

    Args:
        scene_idx (int): Scene to predict TL2LA
        cfg (dict): config 

    Returns:
        pd.DataFrame: prediction based on the scene
    """
    dataset_indices_to_load = set([get_pickled_dataset_index(scene_idx, cfg) for scene_idx in scene_indices])
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_combined_features = executor.map(junctionDataset.get_combined_features, dataset_indices_to_load)
        loaded_scenes = [df.query(f"scene_idx in {scene_indices}") for df in loaded_combined_features]
        
    if not loaded_scenes:
        raise Exception("Nothing to analyze!")
    
    features = pd.concat(loaded_scenes)
    features["turn_type"] = features["lane_id"].apply(mapAPI.get_turn_type)

    # calculation contribution
    frame_contribution = apply_frame_contribution(features)

    scene_prediction = generalize_over_frames(frame_contribution)
    prediction = generalize_over_scenes_by_class_majority(scene_prediction)
    
    # Append intersection information
    prediction = prediction.rename(columns={"sequence_head_lane_id": "lane_id"})
    prediction = prediction.merge(junctionDataset.get_branch_dataset(DATASET_TYPE.MINIMAL), on ="lane_id", how="left")
    return prediction
