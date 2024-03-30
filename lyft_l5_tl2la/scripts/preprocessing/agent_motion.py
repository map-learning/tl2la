# ------------------------------------------------------------------------
# Preprocessing script to calculate the velocity and acceleration of traffic agents (kinematics)
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import re
from time import perf_counter

import pandas as pd
from l5kit.data import ChunkedDataset
from l5kit.data.filter import filter_agents_by_frames
from loguru import logger
from src.common.definitions import DATASET_SIZE, DATASET_TYPE
from src.common.helper import get_filenames, init_dataset, pickle_df
from src.common.utils import (get_movement_frame,
                              merge_agent_lane_mapping_and_branch_frame)
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.dataset_operations.junction_dataset import JunctionDataset
from tqdm import tqdm


def get_scene_entry(path):
    """Returns entry index of scene group for processing
    """
    f_names = get_filenames(path)
    if len(f_names) == 0:
        return 0
    # index of last saved scene group +1
    return int(re.findall(r'\d+', f_names[-1])[0]) + 1


def get_motion_data(junctionDataset: JunctionDataset, zarr_dataset: ChunkedDataset) -> pd.DataFrame:
    """Classifier that applies selected algorithm

    Args:
        junctionDataset (JunctionDataset): Junction Dataset Object 
        zarr_dataset (ChunkedDataset): zarr Dataset
        cfg (dict): config file
    """
    extended_branch_df = junctionDataset.get_branch_dataset(DATASET_TYPE.EXTENDED)
    start_idx = 0

    for dataset_index in tqdm(range(start_idx, DATASET_SIZE), desc="Iterating through preprocessed datasets."):
        # Load agent and traffic light mapping
        agent_mapping_df = junctionDataset.get_agent_mapping(dataset_index)
        agent_mapping_branch_df = merge_agent_lane_mapping_and_branch_frame(
            agent_mapping_df, extended_branch_df)

        scene_indices = agent_mapping_df["scene_idx"].unique()
        motion_dataset_scenes = []
        for scene_idx in scene_indices:
            logger.info(
                f"Motion data for scene: {scene_idx} in dataset {dataset_index}")

            # Get Scene Data
            current_scene = zarr_dataset.get_scene_dataset(
                scene_index=scene_idx)
            current_frames = current_scene.frames
            current_agents = current_scene.agents
            current_agent_frames = filter_agents_by_frames(
                current_frames, current_agents)

            movement_df = get_movement_frame(
                scene_idx, current_frames, current_agent_frames)

            scene_agent_mapping = agent_mapping_branch_df[agent_mapping_branch_df["scene_idx"] == scene_idx]

            scene_motion_df = scene_agent_mapping.merge(
                movement_df.set_index('scene_idx', 'frame_idx', 'agent_id'),
                on=['scene_idx', 'frame_idx', 'agent_id'],
                how='outer'
            )
            scene_motion_df = scene_motion_df.dropna(
                subset=["lane_sequence_id"])

            motion_dataset_scenes.append(scene_motion_df)

        results = pd.concat(motion_dataset_scenes)
        pickle_df(
            f"./dataset/preprocessed_info/agent_motion/{dataset_index}_df_trainfull.pkl", results)


@logger.catch
def main():
    # --------------------------  Logging  -------------------------- #
    log_file_path: str = os.path.join(
        'logs', 'agent_mapping', 'file_{time}.log')
    logger.add(log_file_path, retention=2,
               encoding="utf8", rotation="500 MB", level="TRACE")

    # -----------------  Initialisation of Objects  ------------------ #
    logger.info("Initializing Dataset")
    dataset_type = "trainfull"
    cfg, dm, zarr_dataset = init_dataset(config_name=dataset_type)
    mapAPI = ExtendedMapAPI.from_config(dm, cfg)
    junctionDataset = JunctionDataset(cfg, mapAPI, zarr_dataset)

    # -----------------  Get Motion Data ------------------ #
    logger.info("Get motion data")
    t_start = perf_counter()

    get_motion_data(junctionDataset, zarr_dataset)

    t_stop = perf_counter()
    logger.info(f"Elapsed time: {t_stop-t_start}")


if __name__ == "__main__":
    main()
