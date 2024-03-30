# ------------------------------------------------------------------------
# Implementation of Rejection Method
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import pandas as pd
from scipy import stats

from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.definitions import TL_STATUS, THRESHOLDS_CONSERVATIVE, SCENE_TL_STATUS


def detect_drive_trough(frame_idx: int, velocity: float, dist: float) -> int:
    """Detect if the agent drives throug the intersection based on its velocity and distance to the stop line for the given frame

    Args:
        frame_idx (int): current frame in the scene
        velocity (float): velocity of agent
        dist (float): distance towards stop line

    Returns:
        int: frame idx of drive through. (-1) if no drive through detected
    """
    if velocity >= THRESHOLDS_CONSERVATIVE.V_DRIVE_THROUGH.value/3.6 and dist < THRESHOLDS_CONSERVATIVE.STOP_LINE_DIST.value:
        return frame_idx
    return -1


def is_unallowed_drive_trough(velocity: float, dist: float, tl_state: int, lane_id: str, map_API: ExtendedMapAPI) -> bool:
    """Returns the frame of drive through. -1 if agent does not pass the intersection

    Special case: right turn on red -> only unallowed if drive through at high speed

    Args:
        v (float): velocity
        dist (float): distance

    Returns:
        int: frame idx
    """

    assert tl_state != TL_STATUS.UNKNOWN.value, 'This should not occure and filtered out in previous step'

    # tl state not red but known: allowed to pass intersection
    if tl_state != TL_STATUS.RED.value:
        return False

    # tl state red: forbidden to pass, exept right turn on red!
    is_rigthmost_lane = not map_API.has_adjacent_lane_change_right(lane_id)
    if velocity >= THRESHOLDS_CONSERVATIVE.V_DRIVE_THROUGH.value/3.6 and dist < THRESHOLDS_CONSERVATIVE.STOP_LINE_DIST.value:
        # not right turn
        if not is_rigthmost_lane:
            return True
        # is right turn -> check for higher speed
        if velocity > THRESHOLDS_CONSERVATIVE.V_DRIVE_THROUGH_RIGHT_TURN.value/3.6 and dist < THRESHOLDS_CONSERVATIVE.STOP_LINE_DIST.value:
            return True

    # right turn on red at low speed allowed
    return False


def conservative_data_preprocessing(combined_features: pd.DataFrame, scene_tl_state: pd.DataFrame, extended_branch_df: pd.DataFrame):
    """Data preprocessing for rejection method

    Args:
        combined_features (pd.DataFrame): dataset containing all features of agents, traffic lights and lanes
        scene_tl_state (pd.DataFrame): traffic light state 
        extended_branch_df (pd.DataFrame): junction branch with lane sequence information

    Returns:
        _type_: _description_
    """
    combined_features = combined_features.query("dist <= 5")
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
    combined_features = combined_features.astype({"state_known_start_frame": 'int', "state_known_end_frame": 'int', "state_change_frame": 'int', "interpreted_state": "int"})
    return combined_features



def select_all_drive_throughs(combined_features: pd.DataFrame) -> pd.DataFrame:
    """Selects all agents that performed a drive through in a scene for a specif tl-lane combination. 
    Take the last drive through index since the detection delivers multiple drive throughs for the interval [0 , STOP_LINE_DIST)

    Args:
        combined_features (pd.DataFrame): dataset with al features

    Returns:
        pd.DataFrame: dataset containing drive throughs
    """
    select_columns =  ["scene_idx", "frame_idx", "agent_id", "v", "dist", "junction_id", "branch_id", "sequence_head_lane_id", "tl_id", "tl_status", "state_known_start_frame", "state_known_end_frame", "state_change_frame", "interpreted_state", "drive_trough_idx", "is_unallowed_drive_trough"]
    return combined_features.query(f"drive_trough_idx >= 0").drop_duplicates(subset=["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], keep="last")[select_columns]


def select_unallowed_drive_throughs_const_red(drive_troughs: pd.DataFrame) -> pd.DataFrame:
    """Select all unallowed drive throughs on a const. red traffic light

    Args:
        drive_troughs (pd.DataFrame): dataset with vehicles driving throug an intersection

    Returns:
        pd.DataFrame: dataset with vehicles driving throug an intersection at red light
    """
    drive_trough_on_constant_red = drive_troughs.query(f"tl_status == {TL_STATUS.RED.value} and drive_trough_idx >= 0 and interpreted_state == {SCENE_TL_STATUS.CONST_RED.value} and is_unallowed_drive_trough == True")
    return drive_trough_on_constant_red


def select_unallowed_drive_throughs_before_red_green_change(drive_troughs: pd.DataFrame) -> pd.DataFrame:
    """Select all unallowed drive throughs on before a red to green change (on a red traffic light)

    Args:
        drive_troughs (pd.DataFrame): dataset with vehicles driving throug an intersection

    Returns:
        pd.DataFrame: dataset with vehicles driving throug an intersection at red light
    """
    drive_trough_before_red_green = drive_troughs.query(f"tl_status == {TL_STATUS.RED.value} and drive_trough_idx >= 0 and interpreted_state == {SCENE_TL_STATUS.RED_GREEN.value} and drive_trough_idx <= state_change_frame and is_unallowed_drive_trough == True")
    return drive_trough_before_red_green

def select_unallowed_drive_throughs_after_green_red_change(drive_troughs: pd.DataFrame) -> pd.DataFrame:
    """Select all unallowed drive throughs on after a green to red change (on a red traffic light)

    Args:
        drive_troughs (pd.DataFrame): dataset with vehicles driving throug an intersection

    Returns:
        pd.DataFrame: dataset with vehicles driving throug an intersection at red light
    """
    
    drive_trough_after_green_red = drive_troughs.query(f"tl_status == {TL_STATUS.RED.value} and drive_trough_idx >= 0 and interpreted_state == {SCENE_TL_STATUS.GREEN_RED.value} and drive_trough_idx >= state_change_frame + 15 and is_unallowed_drive_trough == True")
    return drive_trough_after_green_red

def select_unallowed_red_drive_throughs(drive_troughs: pd.DataFrame) -> pd.DataFrame:
    """Select all unallowed drive throughs on red light

    Args:
        drive_troughs (pd.DataFrame): dataset with vehicles driving throug an intersection

    Returns:
        pd.DataFrame: dataset with vehicles driving throug an intersection at red light
    """
    drive_trough_on_constant_red = select_unallowed_drive_throughs_const_red(drive_troughs)
    drive_trough_before_red_green = select_unallowed_drive_throughs_before_red_green_change(drive_troughs)
    drive_trough_after_green_red = select_unallowed_drive_throughs_after_green_red_change(drive_troughs)
    
    red_cases = [drive_trough_on_constant_red, drive_trough_before_red_green, drive_trough_after_green_red]
    # red_cases = [case.drop_duplicates(subset=["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], keep="last") for case in red_cases]
    drive_trough_on_red = pd.concat(red_cases)
    return drive_trough_on_red

def count_drive_throughs(drive_throughs: pd.DataFrame, count_column_name: str) -> pd.DataFrame:
    """Count the number of vehicles driving through an intersection

    Args:
        drive_throughs (pd.DataFrame): dataset with vehicles driving through an intersection
        count_column_name (str): _description_

    Returns:
        pd.DataFrame: drive through count
    """
    return drive_throughs.groupby(by=["sequence_head_lane_id", "tl_id"])["agent_id"].count().reset_index().rename(columns={"agent_id": count_column_name})


def perform_binom_test(all_drives: pd.DataFrame, red_passes: pd.DataFrame, alpha = 0.001,  p_red_drive_through = 0.05) -> pd.DataFrame:
    """Perform binomial test to identify based on red-passes if traffic light is assigned to lane.

    Args:
        all_drives (pd.DataFrame): contains number of all drives for a traffic light lane pair
        red_passes (pd.DataFrame): contains number of red-passes for a traffic light lane pair
        alpha (float): probability that H0 is incorrect
        p_red_drive_through (float): probability to cross intersection unallowed on red traffic light (should be lower or equal to)

    Returns:
        pd.DataFrame: result of binomial test assigning each traffic light lane pair an association (assignment/no assignment)
    """

    tl_lane_stats = all_drives.merge(red_passes, on=["sequence_head_lane_id", "tl_id"], how="left")
    tl_lane_stats = tl_lane_stats.fillna(0).astype({"num_unallowed_red_drive_throughts": int})
    tl_lane_stats["num_red_drive_throughts_percentage"]  = tl_lane_stats["num_unallowed_red_drive_throughts"] / tl_lane_stats["num_drive_throughs"]
    
    # if pvalue <=alpha -> reject H0 -> no assignment between tl-lane pair
    tl_lane_stats["invalid_assoc"] = tl_lane_stats.apply(lambda row: stats.binomtest(k=row["num_unallowed_red_drive_throughts"], n=row["num_drive_throughs"], p=p_red_drive_through, alternative="greater").pvalue <= alpha, axis=1) 
    return tl_lane_stats