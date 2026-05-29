#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

from typing import Any

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.experimental.hide_and_seek.hide_and_seek_module import HideAndSeekModule
from dimos.mapping.costmapper import CostMapper
from dimos.perception.perceive_loop_skill import PerceiveLoopSkill
from dimos.perception.spatial_perception import SpatialMemory
from dimos.robot.unitree.go2.blueprints.agentic._common_agentic import _common_agentic
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.rerun_config import go2_rerun_pubsubs
from dimos.visualization.vis_module import vis_module


def _convert_camera_info(camera_info: Any) -> Any:
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="camera_optical",
    )


def _convert_navigation_costmap(grid: Any) -> Any:
    return grid.to_rerun(
        colormap="Accent",
        z_offset=0.015,
        opacity=0.2,
        background="#484981",
    )


def _static_base_link(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.35, 0.155, 0.2],
            colors=[(0, 255, 127)],
            fill_mode="wireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _go2_hns_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/color_image", name="Camera"),
                rrb.Spatial2DView(origin="world/detection", name="Detection"),
                row_shares=[1, 1],
            ),
            rrb.Spatial3DView(origin="world", name="Map"),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


_rerun_config = {
    "blueprint": _go2_hns_rerun_blueprint,
    "pubsubs": go2_rerun_pubsubs(),
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/navigation_costmap": _convert_navigation_costmap,
    },
    "static": {
        "world/tf/base_link": _static_base_link,
    },
}

_unitree_go2_hns_spatial = autoconnect(
    unitree_go2,
    SpatialMemory.blueprint(),
    PerceiveLoopSkill.blueprint(),
).global_config(n_workers=8)

unitree_go2_hns = autoconnect(
    _unitree_go2_hns_spatial,
    McpServer.blueprint(),
    McpClient.blueprint(model="openai:doubao-seed-2.0-pro"),
    _common_agentic,
    HideAndSeekModule.blueprint(),
    vis_module(viewer_backend=global_config.viewer, rerun_config=_rerun_config),
).remappings(
    [
        (CostMapper, "global_costmap", "raw_global_costmap"),
        (HideAndSeekModule, "global_costmap", "raw_global_costmap"),
        (HideAndSeekModule, "bounded_global_costmap", "global_costmap"),
        (HideAndSeekModule, "cached_global_costmap", "global_costmap"),
    ]
)

__all__ = ["unitree_go2_hns"]
