# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform

from dimos.experimental.hide_and_seek.hide_and_seek_module import HideAndSeekModule
from dimos.experimental.security_demo.security_module import SecurityModule
from dimos.mapping.costmapper import CostMapper
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.patrolling.module import PatrollingModule
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.perception.perceive_loop_skill import PerceiveLoopSkill
from dimos.perception.spatial_perception import SpatialMemory
from dimos.robot.unitree.go2.blueprints.agentic.unitree_go2_hns import (
    _rerun_config,
    unitree_go2_hns,
)
from dimos.visualization.rerun.bridge import TopicTransportPubSub


def test_hns_blueprint_excludes_security_module() -> None:
    modules = {atom.module for atom in unitree_go2_hns.active_blueprints}

    assert HideAndSeekModule in modules
    assert WavefrontFrontierExplorer in modules
    assert SpatialMemory in modules
    assert PerceiveLoopSkill in modules
    assert SecurityModule not in modules


def test_hns_blueprint_republishes_cached_costmap_as_global_costmap() -> None:
    assert (
        unitree_go2_hns.remapping_map[(HideAndSeekModule, "cached_global_costmap")]
        == "global_costmap"
    )


def test_hns_blueprint_routes_bounded_costmap_to_navigation() -> None:
    assert unitree_go2_hns.remapping_map[(CostMapper, "global_costmap")] == "raw_global_costmap"
    assert (
        unitree_go2_hns.remapping_map[(HideAndSeekModule, "global_costmap")] == "raw_global_costmap"
    )
    assert (
        unitree_go2_hns.remapping_map[(HideAndSeekModule, "bounded_global_costmap")]
        == "global_costmap"
    )
    assert (ReplanningAStarPlanner, "global_costmap") not in unitree_go2_hns.remapping_map
    assert (WavefrontFrontierExplorer, "global_costmap") not in unitree_go2_hns.remapping_map
    assert (PatrollingModule, "global_costmap") not in unitree_go2_hns.remapping_map


def test_hns_rerun_config_constructs_blueprint() -> None:
    assert callable(_rerun_config["blueprint"])
    assert _rerun_config["blueprint"]() is not None
    assert _rerun_config["pubsubs"]
    assert "world/camera_info" in _rerun_config["visual_override"]
    assert "world/navigation_costmap" in _rerun_config["visual_override"]

    if platform.system() != "Linux":
        assert any(isinstance(pubsub, TopicTransportPubSub) for pubsub in _rerun_config["pubsubs"])
