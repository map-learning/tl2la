# ------------------------------------------------------------------------
# Preprocessing script to merge the preprocessed datasets
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import re
from time import perf_counter

import pandas as pd
from loguru import logger
from src.common.definitions import DATASET_SIZE
from src.common.helper import get_filenames, init_dataset, pickle_df
from src.common.utils import (calculate_elapsed_time_in_tl_state,
                              calculate_lead_vehicle, calculate_tl_distance,
                              calculate_traffic_density,
                              detect_active_status_change,
                              generate_tl_lane_relation_lookup,
                              get_relation_by_lookuptable)
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.dataset_operations.junction_dataset import JunctionDataset
from tqdm import tqdm


def get_start_idx(path):
    """Get current index to preprocess datasets
    """
    f_names = get_filenames(path)
    if len(f_names) == 0:
        return 0
    # index of last saved scene group +1
    return int(re.findall(r'\d+', f_names[-1])[0]) + 1


def get_features(junctionDataset: JunctionDataset) -> pd.DataFrame:
    """Classifier that applies selected algorithm
    Args:
        junctionDataset (JunctionDataset): Junction Dataset Object 
        zarr_dataset (ChunkedDataset): zarr Dataset
        cfg (dict): config file
    """
    path = "dataset/preprocessed_info/combined_features/"
    start_idx = get_start_idx(path)
    logger.info(f"Starting to process dataset: {start_idx}")
    # Generate traffic light to lane lookup
    relation_lookup = generate_tl_lane_relation_lookup(junctionDataset.get_ground_truth())
    
    for dataset_index in tqdm(range(start_idx, DATASET_SIZE), desc="Iterating through preprocessed datasets."):
        # Load agent motion and traffic light mapping
        motion_df = junctionDataset.get_agent_motion(dataset_index).reset_index(drop=True)
        tl_state_df = junctionDataset.get_tl_mapping(dataset_index).reset_index(drop=True)
        tl_state_df = tl_state_df.drop(["active_faces",	"inactive_faces", "unknown_faces"], axis=1)
        
        
        status_change_df = detect_active_status_change(tl_state_df)

        motion_df = calculate_traffic_density(motion_df)
        motion_df = calculate_tl_distance(junctionDataset.mapAPI, motion_df)
        motion_df = calculate_lead_vehicle(motion_df)

        merged_df = motion_df.merge(tl_state_df.set_index(
            ["scene_idx", "frame_idx"]), on=["scene_idx", "frame_idx"], how="inner")
        
    
        merged_df["relation"] = merged_df.apply(lambda row: get_relation_by_lookuptable(
            row["tl_id"], row["sequence_head_lane_id"], relation_lookup), axis=1)
        merged_df = merged_df[merged_df["relation"] != -2] #remove non existing relations
        
        
        merged_df["time_in_tl_state"] = merged_df.apply(lambda row: calculate_elapsed_time_in_tl_state(
            row["scene_idx"], row["frame_idx"], status_change_df, row["tl_id"]), axis=1)

        merged_df = merged_df.dropna()
        merged_df[["x", "y"]] = pd.DataFrame(merged_df["centroid"].tolist(), index= merged_df.index)
        
        merged_df = merged_df.drop(["lane_sequence_id", "time", "centroid"], axis=1)

        pickle_df(
            f"./dataset/preprocessed_info/combined_features/{dataset_index}_df_trainfull.pkl", merged_df)


@logger.catch
def main():
    # --------------------------  Logging  -------------------------- #
    log_file_path: str = os.path.join(
        'logs', 'feature_generation', 'file_{time}.log')
    logger.add(log_file_path, retention=2,
               encoding="utf8", rotation="500 MB", level="TRACE")

    # -----------------  Initialisation of Objects  ------------------ #
    logger.info("Initializing Dataset")
    dataset_type = "trainfull"
    cfg, dm, zarr_dataset = init_dataset(config_name=dataset_type)
    mapAPI = ExtendedMapAPI.from_config(dm, cfg)
    junctionDataset = JunctionDataset(cfg, mapAPI, zarr_dataset)

    # -----------------  Get Features------------------ #
    logger.info("Get combined features")
    t_start = perf_counter()

    get_features(junctionDataset)

    t_stop = perf_counter()
    logger.info(f"Elapsed time: {(t_stop-t_start)/60}min.")


if __name__ == "__main__":
    main()