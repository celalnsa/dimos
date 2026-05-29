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

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import get_args
from unittest.mock import call

import numpy as np
import pytest

from dimos.experimental.hide_and_seek.hide_and_seek_module import (
    Cv2HogPersonDetector,
    DetectorBackend,
    HideAndSeekModule,
    _create_person_detector,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D


@pytest.fixture()
def hns_module(mocker):
    mocker.patch("dimos.experimental.hide_and_seek.hide_and_seek_module._create_router")
    mocker.patch("dimos.experimental.hide_and_seek.hide_and_seek_module._create_person_detector")

    module = HideAndSeekModule()
    module.config.use_cached_map = False
    module.config.save_map_cache = False
    module.hide_and_seek_state = mocker.MagicMock()
    module.detection = mocker.MagicMock()
    module.goal_request = mocker.MagicMock()
    module.cmd_vel = mocker.MagicMock()
    module.bounded_global_costmap = mocker.MagicMock()
    module.cached_global_costmap = mocker.MagicMock()
    module._planner_spec = mocker.MagicMock()
    module._speak_skill = mocker.MagicMock()
    module._sport_skill = mocker.MagicMock()
    module._sport_skill.execute_sport_command.return_value = "ok"
    module._explorer_spec = mocker.MagicMock()
    module._explorer_spec.explore.return_value = True
    module._explorer_spec.is_exploration_active.return_value = False
    module._router = mocker.MagicMock()
    module._fallback_router = mocker.MagicMock()
    module._detector = mocker.MagicMock()
    module._stop_event.wait = mocker.MagicMock(return_value=False)
    module._costmap_event.wait = mocker.MagicMock(return_value=True)

    yield module

    module.stop()


@pytest.fixture()
def image() -> Image:
    return Image.from_numpy(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture()
def person_detection(image: Image) -> Detection2DBBox:
    return Detection2DBBox(
        bbox=(100.0, 80.0, 260.0, 420.0),
        track_id=1,
        class_id=0,
        confidence=0.95,
        name="person",
        ts=image.ts,
        image=image,
    )


def _costmap() -> OccupancyGrid:
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[0, 0] = 100
    return OccupancyGrid(grid=grid, resolution=0.1)


def _large_costmap() -> OccupancyGrid:
    return OccupancyGrid(grid=np.zeros((25, 25), dtype=np.int8), resolution=1.0)


def test_detector_backend_does_not_expose_mps() -> None:
    assert get_args(DetectorBackend) == ("auto", "yolo-cpu", "hog")


def test_auto_detector_prefers_cpu_yolo(mocker):
    created_devices = []

    class FakeYoloDetector:
        def __init__(self, device):
            created_devices.append(device)

    mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module.YoloPersonDetectorAdapter",
        FakeYoloDetector,
    )

    detector = _create_person_detector("auto", min_confidence=0.0)

    assert isinstance(detector, FakeYoloDetector)
    assert created_devices == ["cpu"]


def test_auto_detector_falls_back_to_hog_when_cpu_yolo_fails(mocker):
    created_devices = []

    class FailingYoloDetector:
        def __init__(self, device):
            created_devices.append(device)
            raise RuntimeError("yolo unavailable")

    class FakeHogDetector:
        def __init__(self, min_confidence):
            self.min_confidence = min_confidence

    mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module.YoloPersonDetectorAdapter",
        FailingYoloDetector,
    )
    mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module.Cv2HogPersonDetector",
        FakeHogDetector,
    )

    detector = _create_person_detector("auto", min_confidence=0.25)

    assert isinstance(detector, FakeHogDetector)
    assert detector.min_confidence == 0.25
    assert created_devices == ["cpu"]


def test_countdown_speaks_each_second(hns_module):
    hns_module._run_countdown(3)

    hns_module._speak_skill.speak.assert_has_calls(
        [
            call("three", blocking=False),
            call("two", blocking=False),
            call("one", blocking=False),
        ]
    )
    assert hns_module._stop_event.wait.call_count == 3


def test_default_countdown_starts_with_english_word(hns_module):
    hns_module._run_countdown(10)

    hns_module._speak_skill.speak.assert_any_call("ten", blocking=False)
    hns_module._speak_skill.speak.assert_any_call("one", blocking=False)


def test_enable_person_deduper_reads_environment(mocker, monkeypatch):
    mocker.patch("dimos.experimental.hide_and_seek.hide_and_seek_module._create_router")
    mocker.patch("dimos.experimental.hide_and_seek.hide_and_seek_module._create_person_detector")
    create_deduper = mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module._create_person_deduper"
    )
    monkeypatch.setenv("ENABLE_PERSON_DEDUPER", "1")

    module = HideAndSeekModule()

    try:
        assert module.config.enable_person_deduper is True
        create_deduper.assert_called_once()
    finally:
        module.stop()


def test_main_loop_maps_before_countdown_when_cache_is_missing(hns_module, mocker):
    events = []
    hns_module._load_cached_costmap = mocker.MagicMock(return_value=None)
    hns_module._scan_for_map = mocker.MagicMock(side_effect=lambda _seconds: events.append("map"))
    hns_module._run_countdown = mocker.MagicMock(
        side_effect=lambda _seconds: events.append("countdown")
    )
    hns_module._search_step = mocker.MagicMock(side_effect=lambda: hns_module._stop_event.set())

    hns_module._main_loop(countdown_seconds=10, max_search_seconds=30.0)

    assert events[:2] == ["map", "countdown"]
    hns_module._load_cached_costmap.assert_called_once()
    hns_module._scan_for_map.assert_called_once_with(hns_module.config.initial_map_scan_seconds)
    hns_module._speak_skill.speak.assert_has_calls(
        [
            call("Scanning the room.", blocking=False),
            call("Exploring the room.", blocking=False),
            call("Ready or not, here I come!", blocking=False),
        ]
    )


def test_main_loop_skips_map_scan_when_cached_costmap_exists(hns_module, mocker):
    events = []
    cached = _costmap()
    hns_module._load_cached_costmap = mocker.MagicMock(return_value=cached)
    hns_module._prime_cached_costmap = mocker.MagicMock(
        side_effect=lambda _costmap: events.append("cache")
    )
    hns_module._scan_for_map = mocker.MagicMock()
    hns_module._run_countdown = mocker.MagicMock(
        side_effect=lambda _seconds: events.append("countdown")
    )
    hns_module._search_step = mocker.MagicMock(side_effect=lambda: hns_module._stop_event.set())

    hns_module._main_loop(countdown_seconds=10, max_search_seconds=30.0)

    assert events[:2] == ["cache", "countdown"]
    hns_module._prime_cached_costmap.assert_called_once_with(cached)
    hns_module._scan_for_map.assert_not_called()
    hns_module._speak_skill.speak.assert_has_calls(
        [
            call("Ready or not, here I come!", blocking=False),
        ]
    )


def test_global_costmap_is_saved_to_cache(hns_module, tmp_path):
    cache_path = tmp_path / "hide_and_seek" / "global_costmap.lcm"
    hns_module.config.save_map_cache = True
    hns_module.config.map_cache_path = str(cache_path)
    hns_module.config.min_cached_costmap_free_cells = 1

    hns_module._on_global_costmap(_costmap())

    decoded = OccupancyGrid.lcm_decode(cache_path.read_bytes())
    assert decoded.width == 20
    assert decoded.height == 20
    assert decoded.free_cells == 399


def test_global_costmap_is_limited_to_start_radius(hns_module):
    hns_module.config.map_max_radius_m = 10.0
    hns_module._mapping_origin_xy = (0.0, 0.0)

    hns_module._on_global_costmap(_large_costmap())

    bounded = hns_module.bounded_global_costmap.publish.call_args[0][0]
    assert bounded.grid[0, 0] == 0
    assert bounded.grid[0, 11] == 100
    hns_module._router.handle_occupancy_grid.assert_called_with(bounded)


def test_mapping_phase_runs_frontier_exploration_after_bootstrap_scan(hns_module, mocker):
    hns_module.config.map_exploration_poll_seconds = 0.0
    hns_module._scan_for_map = mocker.MagicMock()
    hns_module._explorer_spec.is_exploration_active.side_effect = [True, False]

    hns_module._run_mapping_phase()

    hns_module._scan_for_map.assert_called_once_with(hns_module.config.initial_map_scan_seconds)
    hns_module._explorer_spec.explore.assert_called_once()
    hns_module._speak_skill.speak.assert_has_calls(
        [
            call("Scanning the room.", blocking=False),
            call("Exploring the room.", blocking=False),
        ]
    )


def test_mapping_phase_stops_frontier_exploration_on_timeout(hns_module, mocker):
    hns_module.config.initial_map_scan_seconds = 0.0
    hns_module.config.map_exploration_timeout_seconds = 0.0
    hns_module._scan_for_map = mocker.MagicMock()
    hns_module._explorer_spec.is_exploration_active.return_value = True

    hns_module._run_mapping_phase()

    hns_module._explorer_spec.stop_exploration.assert_called_once()


def test_scan_for_map_keeps_latest_costmap_when_no_new_refresh(hns_module, mocker):
    costmap = _costmap()
    hns_module._latest_costmap = costmap
    hns_module._start_scanning_in_place = mocker.MagicMock()
    hns_module._stop_scanning_in_place = mocker.MagicMock()
    hns_module._prime_cached_costmap = mocker.MagicMock()
    hns_module._costmap_event.wait = mocker.MagicMock(return_value=False)

    hns_module._scan_for_map(scan_seconds=1.0)

    hns_module._prime_cached_costmap.assert_called_once_with(costmap)


def test_main_loop_speaks_english_when_search_times_out(hns_module, mocker):
    hns_module._run_countdown = mocker.MagicMock()
    hns_module._scan_for_map = mocker.MagicMock()
    hns_module._search_step = mocker.MagicMock()
    mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module.time",
        SimpleNamespace(monotonic=mocker.MagicMock(side_effect=[0.0, 1.0, 31.0])),
    )

    hns_module._main_loop(countdown_seconds=10, max_search_seconds=30.0)

    hns_module._speak_skill.speak.assert_any_call("I could not find everyone.", blocking=False)


def test_start_hide_and_seek_configures_navigation_and_starts_thread(hns_module, mocker):
    threads = []

    class FakeThread:
        def __init__(self, target, args, daemon, name):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = name
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.started = False

        def is_alive(self):
            return self.started

    mocker.patch(
        "dimos.experimental.hide_and_seek.hide_and_seek_module.threading.Thread", FakeThread
    )

    result = hns_module.start_hide_and_seek(countdown_seconds=1, max_search_seconds=2.0)

    assert result.startswith("Hide and seek started.")
    hns_module._router.reset.assert_called_once()
    hns_module._planner_spec.set_replanning_enabled.assert_called_with(False)
    hns_module._planner_spec.set_safe_goal_clearance.assert_called_once()
    assert hns_module._target_people == 3
    assert hns_module._found_people == 0
    assert len(threads) == 1
    assert threads[0].args == (1, 2.0)
    assert threads[0].daemon is True
    assert threads[0].started is True


def test_search_step_requests_patrol_goal_when_idle(hns_module, image):
    goal = PoseStamped(position=[1, 2, 0], orientation=[0, 0, 0, 1])
    hns_module._router.next_goal.return_value = goal
    hns_module._latest_image = image

    hns_module._search_step()

    hns_module.goal_request.publish.assert_called_once_with(goal)
    assert hns_module._has_active_goal is True


def test_search_step_uses_random_fallback_when_coverage_has_no_goal(hns_module, image, mocker):
    fallback_goal = PoseStamped(position=[3, 4, 0], orientation=[0, 0, 0, 1])
    hns_module._router.next_goal.return_value = None
    hns_module._fallback_router.next_goal.return_value = fallback_goal
    hns_module._latest_image = image
    hns_module._start_scanning_in_place = mocker.MagicMock()

    hns_module._search_step()

    hns_module.goal_request.publish.assert_called_once_with(fallback_goal)
    assert hns_module._has_active_goal is True
    hns_module._start_scanning_in_place.assert_not_called()


def test_search_step_scans_and_checks_image_when_no_patrol_goal(hns_module, image, mocker):
    hns_module._router.next_goal.return_value = None
    hns_module._fallback_router.next_goal.return_value = None
    hns_module._latest_image = image
    hns_module._start_scanning_in_place = mocker.MagicMock()
    hns_module._detector.process_image.return_value = ImageDetections2D(
        image=image,
        detections=[],
    )

    hns_module._search_step()

    hns_module.goal_request.publish.assert_not_called()
    hns_module._start_scanning_in_place.assert_called_once()
    hns_module._detector.process_image.assert_called_once_with(image)
    hns_module._stop_event.wait.assert_called_with(timeout=hns_module.config.search_retry_seconds)


def test_scan_in_place_publishes_continuously(hns_module):
    hns_module._stop_event.wait = hns_module._stop_event.__class__().wait
    hns_module.config.scan_command_period_seconds = 0.01

    hns_module._start_scanning_in_place()
    deadline = time.monotonic() + 1.0
    while hns_module.cmd_vel.publish.call_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    hns_module._stop_scanning_in_place()

    assert hns_module.cmd_vel.publish.call_count >= 2
    first_twist = hns_module.cmd_vel.publish.call_args_list[0][0][0]
    assert first_twist.angular.z == hns_module.config.scan_angular_speed
    assert hns_module.cmd_vel.publish.call_args_list[-1][0][0].is_zero()


def test_search_step_finds_person_and_stops_robot(hns_module, image, person_detection):
    hns_module.config.confirmation_frames = 1
    hns_module._target_people = 1
    hns_module._latest_image = image
    hns_module._latest_pose = PoseStamped(position=[1, 2, 0], orientation=[0, 0, 0, 1])
    hns_module._has_active_goal = True
    hns_module._detector.process_image.return_value = ImageDetections2D(
        image=image,
        detections=[person_detection],
    )

    hns_module._search_step()

    assert hns_module._state == "FOUND"
    hns_module._speak_skill.speak.assert_called_once_with("Found you.", blocking=False)
    hns_module._sport_skill.execute_sport_command.assert_has_calls(
        [
            call("BalanceStand"),
            call("Hello"),
        ]
    )
    hns_module._planner_spec.cancel_goal.assert_called()
    hns_module.goal_request.publish.assert_not_called()
    published_twist = hns_module.cmd_vel.publish.call_args[0][0]
    assert isinstance(published_twist, Twist)
    assert published_twist.is_zero()
    hns_module.detection.publish.assert_called_once()
    assert hns_module._stop_event.is_set()


def test_search_step_continues_until_three_people_are_found(hns_module, image, person_detection):
    hns_module.config.confirmation_frames = 1
    hns_module._latest_image = image
    hns_module._has_active_goal = True

    detected = ImageDetections2D(image=image, detections=[person_detection])
    empty = ImageDetections2D(image=image, detections=[])
    hns_module._detector.process_image.side_effect = [
        detected,
        empty,
        detected,
        empty,
        detected,
    ]

    hns_module._search_step()

    assert hns_module._found_people == 1
    assert not hns_module._stop_event.is_set()

    hns_module._search_step()
    hns_module._search_step()

    assert hns_module._found_people == 2
    assert not hns_module._stop_event.is_set()

    hns_module._search_step()
    hns_module._search_step()

    assert hns_module._found_people == 3
    assert hns_module._stop_event.is_set()
    hns_module._speak_skill.speak.assert_has_calls(
        [
            call("Found you.", blocking=False),
            call("Found you.", blocking=False),
            call("Found you.", blocking=False),
        ]
    )
    assert hns_module._sport_skill.execute_sport_command.call_count == 6


def test_same_visible_person_does_not_count_three_times(hns_module, image, person_detection):
    hns_module.config.confirmation_frames = 1
    hns_module._latest_image = image
    hns_module._has_active_goal = True
    hns_module._detector.process_image.return_value = ImageDetections2D(
        image=image,
        detections=[person_detection],
    )

    hns_module._search_step()
    hns_module._search_step()
    hns_module._search_step()

    assert hns_module._found_people == 1
    assert not hns_module._stop_event.is_set()
    assert hns_module._speak_skill.speak.call_count == 1


def test_deduper_blocks_same_person_after_person_leaves_and_returns(
    hns_module, image, person_detection, mocker
):
    hns_module.config.enable_person_deduper = True
    hns_module.config.confirmation_frames = 1
    hns_module._latest_image = image
    hns_module._has_active_goal = True
    hns_module._person_deduper = SimpleNamespace(
        register_found=mocker.MagicMock(
            side_effect=[
                (True, 0, None),
                (False, 0, 0.93),
            ]
        ),
        stop=mocker.MagicMock(),
    )

    detected = ImageDetections2D(image=image, detections=[person_detection])
    empty = ImageDetections2D(image=image, detections=[])
    hns_module._detector.process_image.side_effect = [detected, empty, detected]

    hns_module._search_step()
    hns_module._search_step()
    hns_module._search_step()

    assert hns_module._found_people == 1
    assert not hns_module._stop_event.is_set()
    assert hns_module._speak_skill.speak.call_count == 1
    assert hns_module._sport_skill.execute_sport_command.call_count == 2
    assert hns_module._person_deduper.register_found.call_count == 2


def test_default_detection_requires_three_consecutive_frames(hns_module, image, person_detection):
    hns_module._target_people = 1
    hns_module._latest_image = image
    hns_module._has_active_goal = True
    hns_module._detector.process_image.return_value = ImageDetections2D(
        image=image,
        detections=[person_detection],
    )

    hns_module._search_step()
    hns_module._search_step()

    assert hns_module._state != "FOUND"
    assert not hns_module._stop_event.is_set()
    hns_module._speak_skill.speak.assert_not_called()
    hns_module._sport_skill.execute_sport_command.assert_not_called()

    hns_module._search_step()

    assert hns_module._state == "FOUND"
    hns_module._speak_skill.speak.assert_called_once_with("Found you.", blocking=False)
    hns_module._sport_skill.execute_sport_command.assert_has_calls(
        [
            call("BalanceStand"),
            call("Hello"),
        ]
    )
    assert hns_module._stop_event.is_set()


def test_stop_hide_and_seek_restores_navigation(hns_module):
    hns_module._latest_pose = PoseStamped(position=[1, 2, 0], orientation=[0, 0, 0, 1])

    result = hns_module.stop_hide_and_seek()

    assert result == "Hide and seek stopped."
    hns_module._planner_spec.set_replanning_enabled.assert_called_with(True)
    hns_module._planner_spec.reset_safe_goal_clearance.assert_called_once()
    hns_module._planner_spec.cancel_goal.assert_called()
    hns_module.goal_request.publish.assert_not_called()
    assert hns_module.cmd_vel.publish.call_args[0][0].is_zero()


def test_cv2_hog_detector_is_cpu_safe_on_blank_image(image):
    detector = Cv2HogPersonDetector()

    detections = detector.process_image(image)

    assert isinstance(detections, ImageDetections2D)
    assert detections.detections == []
