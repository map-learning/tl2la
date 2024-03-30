# ------------------------------------------------------------------------
# Test of Mapping Agent to lane
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

import unittest

from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import init_dataset
from src.common.utils import map_agent_to_lanes


class TestAgentToLaneMapping(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAgentToLaneMapping, self).__init__(*args, **kwargs)

        cfg, dm, self.zarr_dataset = init_dataset("train1")

        self.mapAPI = ExtendedMapAPI.from_config(dm, cfg)

    def test_mapping_1(self):
        """Mapping test for Scene 604 Frame 0.
        """
        scene_idx = 604
        frame_idx = 0
        lane_id_results = ["qoFo"]

        sample_scene = self.zarr_dataset.get_scene_dataset(scene_idx)
        ego_frame = sample_scene.frames[frame_idx]
        mapped_lane_ids = map_agent_to_lanes(
            ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)

        self.assertEqual(mapped_lane_ids, lane_id_results)

    def test_mapping_2(self):
        """Mapping test for Scene 604 Frame 0. 
        """
        scene_idx = 604
        frame_idx = 0
        lane_id_results = ["/Eof"]

        sample_scene = self.zarr_dataset.get_scene_dataset(scene_idx)
        ego_frame = sample_scene.frames[frame_idx]
        mapped_lane_ids = map_agent_to_lanes(
            ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)

        self.assertNotEqual(mapped_lane_ids, lane_id_results)

    def test_mapping_3(self):
        """Mapping test for Scene 23 Frame 99. 
        """
        scene_idx = 23
        frame_idx = 99
        lane_id_results = ['QtPT', 'b+hl', 'xsPT', '89hl']

        sample_scene = self.zarr_dataset.get_scene_dataset(scene_idx)
        ego_frame = sample_scene.frames[frame_idx]
        mapped_lane_ids = map_agent_to_lanes(
            ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)

        self.assertEqual(mapped_lane_ids, lane_id_results)

    def test_mapping_4(self):
        """Mapping test for Scene 99 Frame 23. This scene shows a localisation or map error. 
        Therefore no mapping possible
        """
        scene_idx = 10945
        frame_idx = 221
        lane_id_results = []

        sample_scene = self.zarr_dataset.get_scene_dataset(scene_idx)
        ego_frame = sample_scene.frames[frame_idx]
        mapped_lane_ids = map_agent_to_lanes(
            ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)

        self.assertEqual(mapped_lane_ids, lane_id_results)

    def test_mapping_5(self):
        """Mapping test for Scene 18 Frame 47. The AV should be localized on the small lane tile
        """
        scene_idx = 18
        frame_idx = 47
        lane_id_results = ["DvGy"]

        sample_scene = self.zarr_dataset.get_scene_dataset(scene_idx)
        ego_frame = sample_scene.frames[frame_idx]
        mapped_lane_ids = map_agent_to_lanes(
            ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)

        self.assertEqual(mapped_lane_ids, lane_id_results)


if __name__ == '__main__':
    unittest.main()
