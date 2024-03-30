# ------------------------------------------------------------------------
# Helper Class to Load Junction, branch and Lane Dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import pandas as pd
from src.common.definitions import DATASET_TYPE


class BranchDataset:
    def __init__(self) -> None:
        self.dataset_type = DATASET_TYPE.UNDEFINED
        self.dataset: Optional[pd.DataFrame] = None

    def get_branch_dataset(self, dataset_type: DATASET_TYPE) -> pd.DataFrame:
        """Loads specific branch dataset from file system.

        Args:
            dataset_type (int): which type of dataset to load
        """
        # Check if dataset already loaded -> then it can directly be returned
        if self.dataset is not None and dataset_type.value == self.dataset_type.value:
            return self.dataset
        
        if dataset_type.value == DATASET_TYPE.EXTENDED.value:
            path = "./dataset/preprocessed_info/extended_junction_branch_to_lane.csv"
            return pd.read_csv(path, encoding="ascii")
        elif dataset_type.value == DATASET_TYPE.MINIMAL.value:
            path = "./dataset/preprocessed_info/minimal_junction_branch_to_lane.csv"
            return pd.read_csv(path, encoding="ascii")
        elif dataset_type.value == DATASET_TYPE.TRAIN.value:
            path = "./dataset/preprocessed_info/minimal_junction_branch_to_lane.csv"
            minimal_branch_df = pd.read_csv(path, encoding="ascii")
            TRAIN_JUNCTIONS = minimal_branch_df["junction_id"].unique()[:8]
            return minimal_branch_df[minimal_branch_df["junction_id"].isin(TRAIN_JUNCTIONS)]
        elif dataset_type.value == DATASET_TYPE.TEST.value:
            path = "./dataset/preprocessed_info/minimal_junction_branch_to_lane.csv"
            minimal_branch_df = pd.read_csv(path, encoding="ascii")
            TEST_JUNCTIONS = minimal_branch_df["junction_id"].unique()[8:]
            return minimal_branch_df[minimal_branch_df["junction_id"].isin(TEST_JUNCTIONS)]
        else:
            raise Exception(
                f"Please check given branch type to load: {dataset_type.name}")
