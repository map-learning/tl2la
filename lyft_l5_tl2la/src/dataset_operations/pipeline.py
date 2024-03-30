# ------------------------------------------------------------------------
# Pipline Functions to transform dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import pandas as pd
from typing import List, Tuple
from src.dataset_operations.junction_dataset import JunctionDataset
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import concat_preprocessed_dataframes, calculate_distance
from src.common.definitions import TL_STATUS, THRESHOLDS, SAMPLE_FREQUENCY, STATUS_CHANGE, SCENE_TL_STATUS
import numpy as np
from typing import Optional



def load_preprocessed_datasets(path: str, start_idx: int, num: int, batch_size: int, drop_colums: Optional[list] = None) -> pd.DataFrame:
    """Loads and concats preprocessed dataset

    Args:
        path (str): dataset folder
        start_idx (int): start index
        n (int): number of preprocessed dataset 
        select_colums (list): (Optional) select specific columns of preprocessed dataframe

    Returns:
        pd.DataFrame: concatenated prerpocessed dataset
    """
    BATCH_SIZE = batch_size #5
    end_idx = start_idx+num

    dfs = []
    for i in range((end_idx-start_idx)//BATCH_SIZE):
        offset = i*BATCH_SIZE
        current_df = concat_preprocessed_dataframes(path, start_idx+offset, start_idx+offset+BATCH_SIZE)

        if drop_colums: current_df = current_df.drop(drop_colums, axis=1)
        dfs.append(current_df)
    
    return pd.concat(dfs).reset_index(drop=True)

def load_specific_preprocessed_datasets(path: str, start_idx: int, num: int, batch_size: int,  specific_scenes: np.ndarray, drop_colums: Optional[list] = None) -> pd.DataFrame:
    """Loads and concats preprocessed dataset with specific indices

    Args:
        path (str): dataset folder
        start_idx (int): start index
        n (int): number of preprocessed dataset 
        select_colums (list): (Optional) select specific columns of preprocessed dataframe
        specific_scenes (np.ndarray): (Optional) select specific secenes of preprocessed dataframe

    Returns:
        pd.DataFrame: concatenated preprocessed dataset
    """
    BATCH_SIZE = batch_size #5
    end_idx = start_idx+num

    dfs = []
    for i in range((end_idx-start_idx)//BATCH_SIZE):
        offset = i*BATCH_SIZE
        current_df = concat_preprocessed_dataframes(path, start_idx+offset, start_idx+offset+BATCH_SIZE)
        current_df = current_df[current_df["scene_idx"].isin(specific_scenes)]
        if drop_colums: current_df = current_df.drop(drop_colums, axis=1)
        dfs.append(current_df)
    
    return pd.concat(dfs).reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform dataset cleaning

    Steps:
        1. Drop unknown traffic light frames
        2. Replace value for no relation between traffic light and lane with 0 instead of -1

    Args:
        df (pd.DataFrame): preprocessed data

    Returns:
        pd.DataFrame: cleaned preprocessed data
    """
    # df = df.drop(index = df[df["tl_status"]==TL_STATUS.UNKNOWN.value].index) 
    df["relation"] = df["relation"].replace({-1:0})
    return df

def replace_boolean_values(df: pd.DataFrame)-> pd.DataFrame:
    """Replace boolean values in df with numeric values

    Args:
        df (pd.DataFrame): dataframe to replace boolean values

    Returns:
        pd.DataFrame: dataframe with replaced values
    """
    return df.replace({True:1, False:0})


def filter_out_short_trajectories(df: pd.DataFrame, min_frames: int = 10)-> pd.DataFrame:
    """Filters out agents in scenes which have a trajectory shorter than the given minimum amount of frames.

    Args:
        df (pd.DataFrame): dataframe to filter
        min_frames (int): minimum number of frames

    Returns:
        pd.DataFrame: filtered dataframe
    """
    numer_of_frames_per_agent = df.groupby(["scene_idx", "agent_id"])["frame_idx"].nunique()
    indices_to_drop = numer_of_frames_per_agent[numer_of_frames_per_agent.apply(lambda x: x < min_frames)].index
    grouped_df = df.set_index(["scene_idx", "agent_id"]).drop(index=indices_to_drop)
    df = grouped_df.reset_index()
    return df

def filter_out_agents_off_branch(df: pd.DataFrame)-> pd.DataFrame:
    """Filters out agents that are not on the same branch as the ego vehicle

    Args:
        df (pd.DataFrame): dataframe to filter

    Returns:
        pd.DataFrame: filtered dataframe
    """
    ego_df = df[df["agent_id"]==0]
    ego_branch_per_scene = pd.DataFrame(ego_df.groupby("scene_idx")["branch_id"].unique()).reset_index().explode("branch_id")
    df = df.merge(ego_branch_per_scene, on=["scene_idx", "branch_id"], how="inner")
    return df

def filter_out_far_off_tl(df: pd.DataFrame, range: int = THRESHOLDS.T_TL_DIST.value)-> pd.DataFrame:
    """Filters out traffic lights which are out of range.

    Args:
        df (pd.DataFrame): dataframe to filter
        range (int): maximum range

    Returns:
        pd.DataFrame: filtered dataframe
    """
    return df[df["tl_dist"]<=range].reset_index()

def filter_out_tl_not_off_branch(df: pd.DataFrame, junctionDataset: JunctionDataset, junction_tl_dict: dict)-> pd.DataFrame:
    """Filters out traffic lights that are not belonging to the current branch of the vehicle

    Args:
        df (pd.DataFrame): dataframe to filter
        mapAPI (JunctionDataset): 

    Returns:
        pd.DataFrame: filtered dataframe
    """
    filter_mask = df.apply(lambda row: is_traffic_light_of_junction(row["tl_id"], junctionDataset.get_junction_of_branch(row["branch_id"]), junction_tl_dict), axis=1)
    return df[filter_mask]

def is_traffic_light_of_junction(tl_id: str, junction_id: str, junction_tl_dict: dict)-> bool:
    """Checks if traffic light is part of junction

    Args:
        tl_id (str): id of traffic light
        junction_id (str): junction id
        junction_tl_dict (dict): lookup table

    Returns:
        bool: true if tl belongs to junction
    """
    return tl_id in junction_tl_dict[junction_id]

def filter_frame_tl_states(df: pd.DataFrame, accepted_states: List["int"])-> pd.DataFrame:
    """Filter for specific traffic light states.
    Can be applied on frames and whole scenes.

    Accepted States:
        * RED: 0
        * GREEN: 1

    Args:
        df (pd.DataFrame): dataframe to filter
        accepted_states (list): accepted traffic light states

    Returns:
        pd.DataFrame: filtered dataframe with filtered states
    """
    return df[df["tl_status"].isin(accepted_states)]

def filter_scene_tl_states(df: pd.DataFrame, accepted_states: List["int"])-> pd.DataFrame:
    """Filter for specific traffic light states.
    Can be applied on frames and whole scenes.

    Args:
        df (pd.DataFrame): dataframe to filter
        accepted_states (list): accepted traffic light states

    Returns:
        pd.DataFrame: filtered dataframe with filtered states
    """
    return df[df["scene_tl_state"].isin(accepted_states)]

def filter_lead(df: pd.DataFrame)-> pd.DataFrame:
    """Filter for lead vehicle.
    Can be applied on frames and whole scenes.

    Args:
        df (pd.DataFrame): dataframe to filter

    Returns:
        pd.DataFrame: filtered dataframe 
    """
    if "is_lead" in df.keys(): #applied on frame-based df
        return df[df["is_lead"]>0]
    return df[df["lead_proportion"]>0] #applied on scene-based df


def average_over_scenes(df: pd.DataFrame, group_by: list = ["scene_idx", "agent_id", "sequence_head_lane_id", "tl_id"], cols: dict = {'v': 'mean', 'a':'mean', 'density':'mean', 'tl_status':'mean', 'relation':'mean', 'time_in_tl_state':'mean', 'dist':'mean', 'is_lead':'mean'}) -> pd.DataFrame:
    """Average features over scene

    Args:
        df (pd.DataFrame): dataset to average
        group_by (list): groub by columns

    Returns:
        pd.DataFrame: avg. dataset
    """
    df_scenes = df.groupby(group_by, as_index=False).agg(cols)
    return df_scenes.rename({"v": "v_avg", "a": "a_avg", "is_lead": "lead_proportion", "density": "density_avg"}, axis=1)

def add_start_motion_information(df_scenes: pd.DataFrame, df_frames: pd.DataFrame, num_frames: int = 5)-> pd.DataFrame:
    """Addes start velocity, acceleration and distance of agents trajectory

    Args:
        df (pd.DataFrame): dataframe to append
        num_frames (int, optional): First n frames. Defaults to 5.

    Returns:
        pd.DataFrame: dataframe with added features
    """
    start_features = df_frames.groupby(["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], as_index=False).nth(tuple(range(num_frames))).groupby(["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], as_index=False).mean()
    start_features =  start_features.rename({"v": "v_start", "a":"a_start", "dist": "dist_start"}, axis=1)
    start_features = start_features[["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id", "v_start", "a_start", "dist_start"]]
    df_scenes = df_scenes.merge(start_features, on=["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], how="left")
    return df_scenes


def add_end_motion_information(df_scenes: pd.DataFrame, df_frames: pd.DataFrame, num_frames: int = 5)-> pd.DataFrame:
    """Addes end velocity, acceleration and distance of agents trajectory

    Args:
        df (pd.DataFrame): dataframe to append
        num_frames (int, optional): Last n frames. Defaults to 5.

    Returns:
        pd.DataFrame: dataframe with added features
    """
    end_features = df_frames.groupby(["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], as_index=False).nth(tuple(range(-num_frames, 0))).groupby(["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], as_index=False).mean()
    end_features =  end_features.rename({"v": "v_end", "a":"a_end", "dist": "dist_end"}, axis=1)
    end_features = end_features[["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id", "v_end", "a_end", "dist_end"]]
    df_scenes = df_scenes.merge(end_features, on=["scene_idx", "agent_id", "tl_id", "sequence_head_lane_id"], how="left")
    return df_scenes


def add_drive_through(df: pd.DataFrame, drive_through: pd.DataFrame) -> pd.DataFrame:
    """Appends drive trough information. Chekcs if agent drives in scene trough junction.

    Args:
        df (pd.DataFrame): dataframe to append feature
        drive_trough (pd.DataFrame): dataframe with drive trough information

    Returns:
        pd.DataFrame:  appended dataframe
    """
    return df.merge(drive_through, on=["scene_idx", "agent_id"], how="left").rename(columns={"drive_trough": "drives_across"})

 
def add_tl_state_for_scene(df: pd.DataFrame, tl_state_for_scenes: pd.DataFrame)-> pd.DataFrame:
    """Add traffic light state that has been seen for a traffic light in whole scene

    States:
        * CONST_RED: 0
        * CONST_GREEN: 1
        * RED_GREEN: 2
        * GREEN_RED: 3
        * UNDEFINED: 4

    Args:
        df (pd.DataFrame): dataframe with traffic light states (scene-based)
        tl_state_for_scenes (pd.DataFrame): tl state of scenes (scene-based)

    Returns:
        pd.DataFrame: with interpreted tl_state in scene
    """
    return df.merge(tl_state_for_scenes, on=["tl_id", "scene_idx"], how="left").rename({"interpreted_state":"scene_tl_state"}, axis=1)


def get_acceleration(scene_idx: int, agent_id: int, tl_id: str, state_change_frame: int, frame_features_df: pd.DataFrame, time_interval: int = 50)-> int:
    """Check if agent is accelerating after traffic light state change


    Returns:
        int: flag
    """
    MIN_OVERLAPPING_FRAMES = time_interval//2 
    agent_frame_features_df = frame_features_df[(frame_features_df["scene_idx"]==scene_idx) & (frame_features_df["agent_id"]==agent_id) & (frame_features_df["tl_id"]==tl_id)][["frame_idx", "v", "a", "is_lead", "dist"]]
    relevant_agent_frames = agent_frame_features_df[(agent_frame_features_df["frame_idx"]>state_change_frame) & (agent_frame_features_df["frame_idx"]<(state_change_frame+time_interval))]
    if len(relevant_agent_frames) < MIN_OVERLAPPING_FRAMES:
        return np.nan
    if relevant_agent_frames["is_lead"].sum()==0 or relevant_agent_frames["dist"].mean()< THRESHOLDS.T_STOP_ZONE.value:
        return np.nan
    return relevant_agent_frames["a"].mean() 
    
def add_accelertaion_for_agents(scene_features_df: pd.DataFrame, frame_features: pd.DataFrame, time_interval: int = 50)-> pd.DataFrame:
    """Add flag if agent is accelerating after traffic light turned from red to green.
    
    Args:
        df (pd.DataFrame): dataframe with traffic light states (frame-based)
        df_by_scenes (pd.DataFrame): dataframe averaged over scene

    Returns:
        pd.DataFrame: with detected state in scene
    """
    scene_features_df["acc_green_change"] = 4
    scene_features_df_red_to_green = scene_features_df[scene_features_df["scene_tl_state"]==SCENE_TL_STATUS.RED_GREEN.value].copy()
    scene_features_df_red_to_green["acc_green_change"] =  scene_features_df_red_to_green.apply(lambda row: get_acceleration(row["scene_idx"], row["agent_id"], row["tl_id"], row["state_change_frame"], frame_features, time_interval), axis=1)
    scene_features_df_red_to_green["acc_green_change"] = scene_features_df_red_to_green["acc_green_change"].apply(lambda x: check_accel_threshold(x))
    scene_features_df.loc[scene_features_df_red_to_green.index, "acc_green_change"] = scene_features_df_red_to_green["acc_green_change"]
    del scene_features_df_red_to_green #free memory
    return scene_features_df

def check_accel_threshold(x: np.float)-> int:
    """Check if agent is accelerating

    * true: 1
    * false: 0
    * not seen: 4

    Args:
        x (np.float): acceleration value

    Returns:
        int: returns acceleration check
    """
    if np.isnan(x):
        return 4
    if x > 0.5:
        return 1
    return 0

def add_decelertaion_for_agents(scene_features_df: pd.DataFrame, frame_features: pd.DataFrame, time_interval: int = 50)-> pd.DataFrame:
    """Add flag if agent is accelerating after traffic light turned from red to green.
    
    Args:
        df (pd.DataFrame): dataframe with traffic light states (frame-based)
        df_by_scenes (pd.DataFrame): dataframe averaged over scene

    Returns:
        pd.DataFrame: with detected state in scene
    """
    scene_features_df["dec_red_change"] = 4
    scene_features_df_green_to_red = scene_features_df[scene_features_df["scene_tl_state"]==SCENE_TL_STATUS.GREEN_RED.value].copy()
    scene_features_df_green_to_red["dec_red_change"] =  scene_features_df_green_to_red.apply(lambda row: get_acceleration(row["scene_idx"], row["agent_id"], row["tl_id"], row["state_change_frame"], frame_features, time_interval), axis=1)
    scene_features_df_green_to_red["dec_red_change"] = scene_features_df_green_to_red["dec_red_change"].apply(lambda x: check_decel_threshold(x))
    scene_features_df.loc[scene_features_df_green_to_red.index, "dec_red_change"] = scene_features_df_green_to_red["dec_red_change"]
    del scene_features_df_green_to_red #free memory
    return scene_features_df

def check_decel_threshold(x: np.float)-> int:
    """Check if agent is accelerating

    * true: 1
    * false: 0
    * not seen: 4

    Args:
        x (np.float): acceleration value

    Returns:
        int: returns acceleration check
    """
    if np.isnan(x):
        return 4
    if x < -0.5:
        return 1
    return 0

def calculate_stop_line_distance(mapAPI: ExtendedMapAPI, lane_head_id: str, centroid: np.ndarray) -> np.float:
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

def add_stop_line_distance(df: pd.DataFrame, mapAPI: ExtendedMapAPI)-> pd.DataFrame:
    """Add distance of agent towards stopping line

    Args:drives_across
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df["dist"] = df.apply(lambda row: calculate_stop_line_distance(mapAPI, row["sequence_head_lane_id"], row["centroid"]), axis=1)
    return df

def add_lead_vehicle_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Checks if vehicle is the first vehicle infront of the stopping line.

    Args:
        motion_df (pd.DataFrame): dataframe with motion data and distance information towards lead vehicle

    Returns:
        pd.DataFrame: _description_
    """
    # calculate lead indices
    lead_indices = df.groupby(["scene_idx", "frame_idx", "sequence_head_lane_id"])["dist"].idxmin()
    df["is_lead"] = False
    df.loc[lead_indices.values.tolist(), "is_lead"] = True
    return df


def add_traffic_density(motion_df: pd.DataFrame) -> pd.DataFrame:
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
        red_to_green_change = current_tl_mapping[current_tl_mapping["tl_status"].diff() == 1]
        green_to_red_change = current_tl_mapping[current_tl_mapping["tl_status"].diff() == -1]

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

def add_elapsed_time_in_tl_state(df: pd.DataFrame) -> float:
    """Calcualtes the time elapsed after a state change of a traffic light for whole dataframe

    Args:
        status_change_df (pd.DataFrame): tl state change information 

    Returns:
        float: elapsed time
    """
    status_change_df = detect_active_status_change(df)
    df["time_in_tl_state"] = df.apply(lambda row: calculate_elapsed_time_in_tl_state(row["scene_idx"], row["frame_idx"], status_change_df, row["tl_id"]), axis=1)
    return df

def add_tl_coords(df: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Appends tl coords to dataframe

    Args:
        df (pd.DataFrame): dataframe to append feature

    Returns:
        pd.DataFrame: appended dataframe
    """
    tl_x = mapAPI.get_tl_bounds()["bounds"][:,0,0].tolist()
    tl_y = mapAPI.get_tl_bounds()["bounds"][:,0,1].tolist()
    tl_ids = mapAPI.get_tl_bounds()["ids"]
    d = {"tl_id": tl_ids, "tl_x": tl_x, "tl_y":tl_y}
    tl_coords_df = pd.DataFrame(d)
    return df.merge(tl_coords_df, on="tl_id", how="left")

def add_tl_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Appends current distance between centroid and traffic light. 

    Args:
        df (pd.DataFrame): dataframe to append feature

    Returns:
        pd.DataFrame: appended dataframe
    """
    df["tl_dist"] = df.apply(lambda row: calculate_distance(np.array([row["x"], row["y"]]), np.array([row["tl_x"], row["tl_y"]])), axis=1)
    return df

def add_right_lane_information(df: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Appends right lane information. Checks if lane on which an agent is driving is the rightern-moast lane.

    Args:
        df (pd.DataFrame): dataframe to append feature
        mapAPI (pd.DataFrame): map API

    Returns:
        pd.DataFrame: appended dataframe
    """
    df["is_right_lane"] = df["sequence_head_lane_id"].apply(lambda x: not mapAPI.has_adjacent_lane_change_right(x))
    return df


def get_indices_to_split(features_df: pd.DataFrame, minimal_branch_df: pd.DataFrame) -> Tuple[list, list, pd.DataFrame, pd.DataFrame]:
    """Generate indices to split Inputs and Target into training and test Inputs and Targets

    Args:
        features_df (pd.DataFrame): preprocessed feature dataset
        minimal_branch_df (pd.DataFrame): branch dataset

    Returns:
        Tuple[list, list]: train_indices, test_indices, train_branch_df, test_branch_df
    """
    
    TRAIN_JUNCTIONS = minimal_branch_df["junction_id"].unique()[:6]
    TEST_JUNCTIONS = minimal_branch_df["junction_id"].unique()[6:]

    train_branch_df = minimal_branch_df[minimal_branch_df["junction_id"].isin(TRAIN_JUNCTIONS)]
    test_branch_df = minimal_branch_df[minimal_branch_df["junction_id"].isin(TEST_JUNCTIONS)]

    train_indices = features_df[features_df["sequence_head_lane_id"].isin(train_branch_df["lane_id"].unique())].index.tolist()
    test_indices = features_df[features_df["sequence_head_lane_id"].isin(test_branch_df["lane_id"].unique())].index.tolist()

    return train_indices, test_indices, train_branch_df, test_branch_df

def split_train_test(X: pd.DataFrame, y: pd.Series, train_indices: list, test_indices: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and test data

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): association label
        train_indices (list): indices of training dataset
        test_indices (list): indices of test dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test 
    """

    X_train = X[X.index.isin(train_indices)]
    X_test = X[X.index.isin(test_indices)]
    assert set(X_train.index).intersection(set(X_test.index)) == set()

    y_train = y[y.index.isin(X_train.index)]
    y_test = y[y.index.isin(X_test.index)]
    assert set(y_train.index).intersection(set(y_test.index)) == set()

    return X_train, X_test, y_train, y_test

def split_features_train_test(features: pd.DataFrame, train_indices: list, test_indices: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset features into training and test features

    Args:
        features (pd.DataFrame): features
        train_indices (list): indices for training set
        test_indices (list):  indices for test set

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: features for training and for testing
    """

    features_train = features[features.index.isin(train_indices)]
    features_test = features[features.index.isin(test_indices)]
    assert set(features_train.index).intersection(set(features_test.index)) == set()


    return features_train, features_test

def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Copy dataframe

    Args:
        df (pd.DataFrame): dataframe to copy

    Returns:
        pd.DataFrame: copy of dataframe
    """
    return df.copy()



################ PIPELINES ################

def pipeline_dt_0(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline of Decision tree.
    Frame based decision tree.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """
    df =  (
        df
        .pipe(clean_dataset)
        .pipe(filter_lead)
        .pipe(replace_boolean_values)
    )
    return df


def pipeline_dt_1(df: pd.DataFrame, drive_through: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Pipeline of Decision tree.
    Scene based decision tree.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """

    df_scenes = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_short_trajectories)
        .pipe(copy_df)
        .pipe(average_over_scenes)
        .pipe(filter_frame_tl_states, accepted_states=[TL_STATUS.RED.value, TL_STATUS.GREEN.value])
        .pipe(filter_lead)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=5)
        .pipe(add_end_motion_information,df_frames=df, num_frames=5)
        .pipe(replace_boolean_values)
    )
    del df #delete deepcopy to free memory
    return df_scenes
  
        
        
def pipeline_dt_1_2(df: pd.DataFrame, drive_through: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Pipeline of Decision tree.
    Scene based decision tree. Only creates association to stationary traffic lights.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """

    df_scenes = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_short_trajectories)
        .pipe(add_tl_coords,  mapAPI=mapAPI)
        .pipe(add_tl_distance)
        .pipe(filter_out_far_off_tl)
        .pipe(copy_df)
        .pipe(average_over_scenes)
        .pipe(filter_frame_tl_states, accepted_states=[TL_STATUS.RED.value, TL_STATUS.GREEN.value])
        .pipe(filter_lead)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=10)
        .pipe(add_end_motion_information,df_frames=df, num_frames=10)
        .pipe(replace_boolean_values)
    )
    del df #delete deepcopy to free memory
    return df_scenes

def pipeline_dt_2(df: pd.DataFrame, drive_through: pd.DataFrame, tl_state_scene: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Pipeline of Decision
    Scene based decision tree. Only creates association to stationary traffic lights.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """

    df_scenes = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_short_trajectories, min_frames=60)
        .pipe(copy_df)
        .pipe(average_over_scenes, cols={'v': 'mean', 'a':'mean', 'density':'mean', 'relation':'mean', 'is_lead':'mean'})
        .pipe(add_tl_state_for_scene, tl_state_for_scenes=tl_state_scene)
        .pipe(add_accelertaion_for_agents, frame_features=df, time_interval=60)
        .pipe(filter_lead)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=10)
        .pipe(add_end_motion_information,df_frames=df, num_frames=10)
        .pipe(replace_boolean_values)
    )
    del df #delete deepcopy to free memory
    return df_scenes


def pipeline_dt_3(df: pd.DataFrame, drive_through: pd.DataFrame, tl_state_scene: pd.DataFrame, mapAPI: ExtendedMapAPI, junctionDataset: JunctionDataset, junction_tl_dict: dict) -> pd.DataFrame:
    """Pipeline of Decision tree V3
    Scene based decision tree. Only creates association to stationary traffic lights.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """
    df = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_short_trajectories, min_frames=50)
        .pipe(filter_out_tl_not_off_branch, junctionDataset, junction_tl_dict)
        
    )
    df_scenes = (
        df
        .pipe(average_over_scenes, cols={'v': 'mean', 'a':'mean', 'density':'mean', 'relation':'mean', 'is_lead':'mean'})
        .pipe(add_tl_state_for_scene, tl_state_for_scenes=tl_state_scene)
        .pipe(add_accelertaion_for_agents, frame_features=df, time_interval=50)
        .pipe(filter_lead)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=10)
        .pipe(add_end_motion_information,df_frames=df, num_frames=10)
        .pipe(replace_boolean_values)
    )
    del df #delete deepcopy to free memory
    return df_scenes

def pipeline_dt_4(df: pd.DataFrame, drive_through: pd.DataFrame, tl_state_scene: pd.DataFrame, mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Pipeline of Decision tree.
    Scene based decision tree. Only creates association to stationary traffic lights.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """

    df = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_short_trajectories, min_frames=50)
        .pipe(filter_lead)
    )
    df_scenes = (df
        .pipe(average_over_scenes, cols={'v': 'mean', 'a':'mean', 'density':'mean', 'relation':'mean', 'is_lead':'mean'})
        .pipe(add_tl_state_for_scene, tl_state_for_scenes=tl_state_scene)
        .pipe(add_accelertaion_for_agents, frame_features=df, time_interval=20)
        .pipe(add_decelertaion_for_agents, frame_features=df, time_interval=20)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=10)
        .pipe(add_end_motion_information,df_frames=df, num_frames=10)
        .pipe(replace_boolean_values)
    )
    return df_scenes


def pipeline_dt_5(df: pd.DataFrame, drive_through: pd.DataFrame, tl_state_scene: pd.DataFrame, mapAPI: ExtendedMapAPI,  junctionDataset: JunctionDataset, junction_tl_dict: dict) -> pd.DataFrame:
    """Pipeline of Decision tree V1
    Scene based decision tree. Only creates association to stationary traffic lights.

    Args:
        df (pd.DataFrame): loaded preprocessed dataset

    Returns:
        pd.DataFrame: transformed dataset
    """

    df = (
        df
        .pipe(clean_dataset)
        .pipe(filter_out_agents_off_branch)
        .pipe(filter_out_tl_not_off_branch, junctionDataset, junction_tl_dict)
        .pipe(filter_out_short_trajectories, min_frames=60)
        .pipe(copy_df)
       
    )
    df_scenes = (df
        .pipe(average_over_scenes, cols={'v': 'mean', 'a':'mean', 'density':'mean', 'relation':'mean', 'is_lead':'mean'})
        .pipe(add_tl_state_for_scene, tl_state_for_scenes=tl_state_scene)
        .pipe(filter_lead)
        .pipe(filter_scene_tl_states, accepted_states=[SCENE_TL_STATUS.CONST_GREEN.value, SCENE_TL_STATUS.CONST_RED.value, SCENE_TL_STATUS.GREEN_RED.value, SCENE_TL_STATUS.RED_GREEN.value])
        .pipe(add_accelertaion_for_agents, frame_features=df, time_interval=20)
        .pipe(add_decelertaion_for_agents, frame_features=df, time_interval=20)
        .pipe(add_right_lane_information, mapAPI=mapAPI)
        .pipe(add_drive_through, drive_through=drive_through)
        .pipe(add_start_motion_information, df_frames=df, num_frames=10)
        .pipe(add_end_motion_information,df_frames=df, num_frames=10)
        .pipe(replace_boolean_values)
    )
    return df_scenes

