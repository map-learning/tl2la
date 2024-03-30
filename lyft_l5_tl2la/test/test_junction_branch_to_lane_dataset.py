# ------------------------------------------------------------------------
# Junction Dataset Sanity Checks
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import unittest

from src.dataset_operations.extended_map_api import ExtendedMapAPI
import pandas as pd
from src.common.helper import init_dataset


class TestJunctionBranchToLaneMapping(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestJunctionBranchToLaneMapping, self).__init__(*args, **kwargs)

        preprocessed_data_path = "./dataset/preprocessed_info/junction_branch_to_lane.csv"
        self.branches_df = pd.read_csv(
            preprocessed_data_path, encoding="ascii")

        cfg, dm, self.zarr_dataset = init_dataset("train1")

        self.mapAPI = ExtendedMapAPI.from_config(dm, cfg)

    def testIfLaneIdsExist(self):
        for _, lane_id in self.branches_df["lane_id"].items():
            lane_element = self.mapAPI.get_element_by_id(lane_id)

            is_lane_element = self.mapAPI.element_of_type(lane_element, "lane")
            self.assertTrue(is_lane_element,  msg=f"Element id: {lane_id}")

    def testIfJunctionIdsExist(self):
        for _, junction_id in self.branches_df["junction_id"].items():
            junction_element = self.mapAPI.get_element_by_id(junction_id)

            is_junction_element = self.mapAPI.element_of_type(
                junction_element, "junction")
            self.assertTrue(is_junction_element,
                            msg=f"Element id: {junction_id}")


if __name__ == '__main__':
    unittest.main()
