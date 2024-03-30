# ------------------------------------------------------------------------
# Helper functions to perform dataset transformation and aggregation of 
# Lyft Dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

from src.dataset_operations.junction_dataset import JunctionDataset
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.definitions import DATASET_SIZE
from src.common.helper import load_pickled_df, pickle_df
from src.dataset_operations.pipeline import load_preprocessed_datasets

from src.common.definitions import SCENE_TL_STATUS, DATASET_TYPE

def generate_lane_tl_relation(mapAPI: ExtendedMapAPI) -> dict:
    """Generate the lane to traffic light relation

    Args:
        mapAPI (ExtendedMapAPI): map api

    Returns:
        dict: lane to traffic light association
    """
    lane_tls_dict = dict()
    lane_ids: list = mapAPI.bounds_info["lanes"]["ids"]

    for lane_id in lane_ids:
        lane_traffic_controll_ids = mapAPI.get_lane_traffic_control_ids(
            lane_id)

        # list of traffic controll element of type traffic light next to a lane
        lane_traffic_lights = set()

        for control_id in lane_traffic_controll_ids:
            conrol_element = mapAPI.get_element_by_id(control_id)
            if mapAPI.is_traffic_light(conrol_element):
                lane_traffic_lights = lane_traffic_lights.union(
                    set([control_id]))

        if len(lane_traffic_lights) != 0:
            lane_tls_dict[lane_id] = lane_traffic_lights

    return lane_tls_dict


def generate_predecessor_successor_lookup(mapAPI: ExtendedMapAPI) -> pd.DataFrame:
    """Generate Lane Frame Lookup

    Args:
        mapAPI (ExtendedMapAPI): _description_

    Returns:
        pd.DataFrame: _description_
    """
    lanes = mapAPI.get_elements_from_layer("lane")
    lane_ids = mapAPI.get_element_ids_from_layer("lane")

    # initialize dict
    lane_dict = dict()
    for lane_id in lane_ids:
        lane_dict[lane_id] = {
            'successor': set(), "predecessor": set()}  # initialize keys

    for lane in lanes:
        current_lane_id = mapAPI.id_as_str(lane.id)
        lane_ids_ahead = [mapAPI.id_as_str(id)
                          for id in lane.element.lane.lanes_ahead]

        lane_dict[current_lane_id]["successor"] = set(
            lane_ids_ahead)  # set successing lanes

        for ahead_lane_id in lane_ids_ahead:  # set predecessing lanes
            lane_dict[ahead_lane_id]["predecessor"] = lane_dict[ahead_lane_id]["predecessor"].union(
                set([current_lane_id]))

    predessors = []
    successors = []
    lane_dict_values = lane_dict.values()
    for v in lane_dict_values:
        predessors.append(list(v["predecessor"]))
        successors.append(list(v["successor"]))

    return pd.DataFrame({"lane_id": lane_dict.keys(), "predecessor": predessors, "successor": successors})


def generate_target_label(junctionDataset: JunctionDataset, combinations: pd.DataFrame) -> list:
    """Generate target vector

    Args:
        junctionDataset (JunctionDataset): junction dataset
        combinations (pd.DataFrame): traffic light to sequence head lane ids

    Returns:
        list: calssification labels
    """
    target_vector = []
    for idx, row in combinations.iterrows():
        tl_id = row["tl_id"]
        lane_id = row["lane_id"]

        if tl_id not in junctionDataset.tl_lane_relations.keys():
            target_vector.append(-1)  # relation not available in semantic map
        elif lane_id in junctionDataset.tl_lane_relations[tl_id]:
            target_vector.append(1)
        else:
            target_vector.append(0)
    return target_vector


def generate_target_association(mapAPI: ExtendedMapAPI, junctionDataset: JunctionDataset) -> pd.DataFrame:
    """Generate target classification dataset of all active traffic lights that have been recorded in all scenes.

    Args:
        mapAPI (ExtendedMapAPI): map api
        junctionDataset (JunctionDataset): junction dataset

    Returns:
        pd.DataFrame: target dataset with traffic ligth to lane relations and classification label (1/0/-1)
    """
    extended_branch_df = junctionDataset.get_branch_dataset(DATASET_TYPE.EXTENDED)
    targets = []
    for dataset_index in tqdm(range(DATASET_SIZE), desc="Iterating through preprocessed datasets."):
        tl_mapping_df = junctionDataset.get_tl_mapping(dataset_index)
        agent_mapping_df = junctionDataset.get_agent_mapping(dataset_index)

        ego_mapping_df = agent_mapping_df[agent_mapping_df["agent_id"] == 0].explode(
            "lane_ids").rename(columns={"lane_ids": "lane_id"}).drop_duplicates()

        # filter ego agent mapping for agent on branch lane sequences
        ego_mapping_branch_df = ego_mapping_df.join(
            extended_branch_df.set_index('lane_id'), on=['lane_id'])
        ego_mapping_branch_df = ego_mapping_branch_df.dropna()
        ego_mapping_branch_df = ego_mapping_branch_df.drop_duplicates()
        scene_indices = ego_mapping_branch_df["scene_idx"].unique()

        for scene_idx in scene_indices:
            ego_mapping_current_scene = ego_mapping_branch_df[
                ego_mapping_branch_df["scene_idx"] == scene_idx]
            tl_mapping_current_scene = tl_mapping_df[tl_mapping_df["scene_idx"] == scene_idx]
            unique_branches = ego_mapping_current_scene["branch_id"].unique()

            for branch_id in unique_branches:
                ego_mapping_current_scene_and_branch = ego_mapping_current_scene[
                    ego_mapping_current_scene["branch_id"] == branch_id]
                sequence_head_lane_id_of_branch = extended_branch_df[extended_branch_df[
                    "branch_id"] == branch_id]["sequence_head_lane_id"].tolist()

                start_frame = ego_mapping_current_scene_and_branch["frame_idx"].min(
                )
                start_end = ego_mapping_current_scene_and_branch["frame_idx"].max(
                )
                frame_interval = [
                    start_frame, start_end]

                active_tl_mapping_current_scene = tl_mapping_current_scene[((
                    tl_mapping_current_scene["tl_status"] == 0) | (tl_mapping_current_scene["tl_status"] == 1)) & (tl_mapping_current_scene["frame_idx"].isin(
                        list(range(*frame_interval))))]

                if len(active_tl_mapping_current_scene) < 1:
                    continue

                active_tl_ids = active_tl_mapping_current_scene["tl_id"].unique(
                )
                junction_id = junctionDataset.get_junction_of_branch(branch_id)
                active_tl_ids = [
                    id for id in active_tl_ids if mapAPI.tl_of_junction(id, junction_id)]

                combinations = list(itertools.product(
                    active_tl_ids, sequence_head_lane_id_of_branch))

                for c in combinations:
                    if c not in targets:
                        targets.append(c)

    target_df = pd.DataFrame(
        targets, columns=["tl_id", "lane_id"])
    target_df["relation"] = generate_target_label(junctionDataset, target_df)

    return target_df

def generate_drive_trough_table(mapAPI: ExtendedMapAPI, junctionDataset: JunctionDataset) -> pd.DataFrame:
    """Generate table with drive trough junction description. 
    For every scene and agent.

    Args:
        mapAPI (ExtendedMapAPI): map api
        junctionDataset (JunctionDataset): junction dataset

    Returns:
        pd.DataFrame: Drive trough table
    """
    minimal_branch_df = junctionDataset.get_branch_dataset(DATASET_TYPE.MINIMAL)
    branch_lanes = minimal_branch_df["lane_id"]
    drive_trough_dfs = []
    for dataset_index in tqdm(range(DATASET_SIZE), desc="Iterating through preprocessed datasets."):
        agent_mapping = junctionDataset.get_agent_mapping(dataset_index)

       
        agent_mapping = agent_mapping.explode("lane_ids").rename(columns={"lane_ids":"lane_id"})
        agent_mapping = agent_mapping.drop_duplicates()
        mapped_lanes = agent_mapping.groupby(["scene_idx", "agent_id"])["lane_id"].unique()
        no_mapping_indices = mapped_lanes[mapped_lanes.apply(lambda x: len(x) == 1 and pd.isna(x).any())].index
        reduced_mapped_lanes = mapped_lanes.drop(index=no_mapping_indices)
        reduced_mapped_lanes = reduced_mapped_lanes.reset_index()
        minimal_branch_df = junctionDataset.get_branch_dataset(DATASET_TYPE.MINIMAL)
        branch_lanes = minimal_branch_df["lane_id"]
        intersecting_mapped_lanes = reduced_mapped_lanes["lane_id"].apply(lambda x: set(x).intersection(set(branch_lanes)))
        successor_intersecting_mapped_lanes = intersecting_mapped_lanes.apply(lambda x: [mapAPI.get_lane_successors(lane) for lane in x])

        reduced_mapped_lanes["successors"] = successor_intersecting_mapped_lanes
        reduced_mapped_lanes["successors"] = reduced_mapped_lanes["successors"].apply(lambda x: list(itertools.chain.from_iterable(x)))
        reduced_mapped_lanes["drive_trough"] = reduced_mapped_lanes.apply(lambda x: any(item in x["lane_id"] for item in x["successors"]), axis=1)
        drive_trough_dfs.append(reduced_mapped_lanes)
    
        pickle_df(f"dataset/preprocessed_info/drive_trough/{dataset_index}_df_trainfull.pkl", reduced_mapped_lanes)
    


def generate_traffic_light_scene_table() -> pd.DataFrame:
    traffic_light_mapping_dfs = []
    path = "dataset/preprocessed_info/tl_mapping/"
    start_idx = 0
    step = 20
    batch_size = 5
    for dataset_index in tqdm(range(start_idx, DATASET_SIZE, step), desc="Iterating through preprocessed datasets."):
        tl_mapping_df = load_preprocessed_datasets(path, dataset_index, step, batch_size).drop(columns=["active_faces", "inactive_faces", "unknown_faces"])
        tl_mapping_df = tl_mapping_df.sort_values(["tl_id", "scene_idx", "frame_idx"])

        # 1. Aggregate
        tl_mapping_scene = tl_mapping_df.groupby(["tl_id", "scene_idx"], as_index=False).agg({'tl_status': 'mean'}).copy()

        # 2. Trivial Assignment of constant states over complete scene
        tl_mapping_scene_const_undefined = tl_mapping_scene[tl_mapping_scene["tl_status"]==4]
        tl_mapping_scene.loc[tl_mapping_scene_const_undefined.index, "interpreted_state"] = SCENE_TL_STATUS.UNDEFINED.value

        # 3. Trivial Assignment of constant states over complete scene
        tl_mapping_df_without_unknown = tl_mapping_df[tl_mapping_df["tl_status"]!=4].copy()
        tl_mapping_df_without_unknown = tl_mapping_df_without_unknown.sort_values(["tl_id", "scene_idx", "frame_idx"])

        tl_mapping_scene_without_unknown  = tl_mapping_df_without_unknown.groupby(["tl_id", "scene_idx"], as_index=False).agg({'tl_status': 'mean'})
        tl_mapping_scene_without_unknown["interpreted_state"] = tl_mapping_scene_without_unknown["tl_status"].apply(lambda x: int(x) if (x == SCENE_TL_STATUS.CONST_RED.value or x==SCENE_TL_STATUS.CONST_GREEN.value) else np.nan)

        start_frame_df = tl_mapping_df_without_unknown.groupby(["tl_id", "scene_idx"], as_index=False)["frame_idx"].min().rename({"frame_idx":"state_known_start_frame"}, axis=1)
        end_frame_df = tl_mapping_df_without_unknown.groupby(["tl_id", "scene_idx"],  as_index=False)["frame_idx"].max().rename({"frame_idx":"state_known_end_frame"},  axis=1)
        
        tl_mapping_scene_without_unknown = tl_mapping_scene_without_unknown.merge(start_frame_df, on=["tl_id", "scene_idx"], how="left")
        tl_mapping_scene_without_unknown = tl_mapping_scene_without_unknown.merge(end_frame_df, on=["tl_id", "scene_idx"], how="left")

        red_green_change = tl_mapping_df[tl_mapping_df["tl_status"].diff().abs() == 1]#detect red/green change
        red_green_change= red_green_change.drop_duplicates(subset=["scene_idx", "tl_id"], keep="first") #remove tl_changes more than once between red and 
        red_green_change["interpreted_state"] = red_green_change["tl_status"].apply(lambda x: SCENE_TL_STATUS.RED_GREEN.value if x==1 else SCENE_TL_STATUS.GREEN_RED.value)
        missing_states = tl_mapping_scene_without_unknown[tl_mapping_scene_without_unknown["interpreted_state"].isna()]
        red_green_change_reduced = red_green_change[(red_green_change["scene_idx"].isin(missing_states["scene_idx"].unique())) & (red_green_change["tl_id"].isin(missing_states["tl_id"].unique()))]
        red_green_change_reduced = red_green_change_reduced.rename(columns={"frame_idx":"state_change_frame"}).drop("tl_status", axis=1)

        tl_mapping_scene_without_unknown = tl_mapping_scene_without_unknown.merge(red_green_change_reduced, on=["tl_id", "scene_idx"], how="left")
        tl_mapping_scene_without_unknown["interpreted_state"] = tl_mapping_scene_without_unknown["interpreted_state_x"].fillna(tl_mapping_scene_without_unknown["interpreted_state_y"])
        tl_mapping_scene_without_unknown = tl_mapping_scene_without_unknown.drop(columns=["interpreted_state_x", "interpreted_state_y"], axis=1)

        tl_mapping_scene_unknown = tl_mapping_scene[tl_mapping_scene["tl_status"]==4 ].copy()
        tl_mapping_scene_unknown["state_known_start_frame"] =-1
        tl_mapping_scene_unknown["state_known_end_frame"] =-1
        tl_mapping_scene_unknown["state_change_frame"] =-1
        tl_mapping_scene_unknown =  tl_mapping_scene_unknown[["tl_id", "scene_idx", "tl_status","state_known_start_frame", "state_known_end_frame", "state_change_frame", "interpreted_state"]]
        tl_mapping_scene_total = pd.concat([tl_mapping_scene_without_unknown,tl_mapping_scene_unknown] , ignore_index=True)
        del tl_mapping_scene_unknown

        tl_mapping_scene_total = tl_mapping_scene_total.sort_values(["tl_id", "scene_idx"])
        tl_mapping_scene_total["state_change_frame"] = tl_mapping_scene_total["state_change_frame"].fillna(-1)
        tl_mapping_scene_total = tl_mapping_scene_total[ (tl_mapping_scene_total["interpreted_state"].notna())]
        tl_mapping_scene_total = tl_mapping_scene_total.astype({"scene_idx":"int", "tl_status": "int", "state_change_frame": "int", "interpreted_state":"int"})
        tl_mapping_scene_total = tl_mapping_scene_total.drop(columns=["tl_status"])
        traffic_light_mapping_dfs.append(tl_mapping_scene_total)
    
    return pd.concat(traffic_light_mapping_dfs)



def filter_ego_features():
    """Filter complete features of all agents by ego vehicle
    """
    res = [] 
    for i, dataset_index in tqdm(enumerate(range(DATASET_SIZE)), desc="Iterating through preprocessed datasets."):
        features_df = load_pickled_df(f"dataset/preprocessed_info/combined_features/{dataset_index}_df_trainfull.pkl")
        ego_features_df = features_df[features_df["agent_id"] == 0] 
        res.append(ego_features_df)
        if (i % 100) == 99:
            df = pd.concat(res)
            df = df.drop_duplicates()
            res = []
            pickle_df(f"dataset/preprocessed_info/ego_features/{dataset_index}_df_trainfull.pkl", df) 