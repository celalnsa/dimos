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

from collections.abc import Callable
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal, Protocol

from dimos_lcm.std_msgs import Bool, String
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.skills.speak_skill_spec import SpeakSkillSpec
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT, DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.patrolling.constants import EXTRA_CLEARANCE
from dimos.navigation.patrolling.create_patrol_router import PatrolRouterName, create_patrol_router
from dimos.navigation.patrolling.routers.patrol_router import PatrolRouter
from dimos.navigation.replanning_a_star.module_spec import ReplanningAStarPlannerSpec
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

State = Literal["IDLE", "COUNTDOWN", "MAPPING", "SEARCHING", "FOUND"]
DetectorBackend = Literal["auto", "yolo-cpu", "hog"]
PersonDeduperBackend = Literal["torchreid", "mobileclip"]

_COUNTDOWN_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_COUNTDOWN_TEENS = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_COUNTDOWN_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


class PersonDetector(Protocol):
    def process_image(self, image: Image) -> ImageDetections2D: ...

    def stop(self) -> None: ...


class PersonDeduper(Protocol):
    def register_found(self, detection: Detection2DBBox) -> tuple[bool, int, float | None]: ...

    def reset(self) -> None: ...

    def stop(self) -> None: ...


class SportCommandSkillSpec(Spec, Protocol):
    def execute_sport_command(self, command_name: str) -> str: ...


class FrontierExplorerSpec(Spec, Protocol):
    def explore(self) -> bool: ...

    def stop_exploration(self) -> bool: ...

    def is_exploration_active(self) -> bool: ...


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ""}:
        return False

    logger.warning("Ignoring invalid boolean environment value", name=name, value=value)
    return default


class HideAndSeekModuleConfig(ModuleConfig):
    confirmation_frames: int = 3
    detector_backend: DetectorBackend = "auto"
    detector_min_confidence: float = 0.0
    min_detection_area_px: float = 1000.0
    enable_person_deduper: bool = Field(default_factory=lambda: _env_bool("ENABLE_PERSON_DEDUPER"))
    person_deduper_backend: PersonDeduperBackend = "torchreid"
    person_deduper_similarity_threshold: float = 0.72
    person_deduper_padding_px: int = 0
    person_deduper_max_embeddings_per_person: int = 8
    use_cached_map: bool = True
    save_map_cache: bool = True
    map_cache_path: str | None = ".cache/hide_and_seek/global_costmap.lcm"
    min_cached_costmap_free_cells: int = 25
    map_cache_save_interval_seconds: float = 5.0
    use_frontier_mapping: bool = True
    map_max_radius_m: float | None = 10.0
    map_exploration_timeout_seconds: float = 90.0
    map_exploration_poll_seconds: float = 0.5
    scan_angular_speed: float = 0.35
    scan_command_period_seconds: float = 0.1
    initial_map_scan_seconds: float = 18.0
    map_refresh_wait_seconds: float = 2.0
    search_retry_seconds: float = 2.0
    no_patrol_goal_log_seconds: float = 2.0


class YoloPersonDetectorAdapter:
    """Lazy wrapper around the stronger YOLO person detector."""

    def __init__(self, device: Literal["cpu"]) -> None:
        from dimos.perception.detection.detectors.person.yolo import YoloPersonDetector

        self._device = device
        self._detector = YoloPersonDetector(device=device)

    def process_image(self, image: Image) -> ImageDetections2D:
        return self._detector.process_image(image)

    def stop(self) -> None:
        self._detector.stop()


class Cv2HogPersonDetector:
    """CPU-only OpenCV person detector used as a macOS-safe fallback."""

    def __init__(self, min_confidence: float = 0.0) -> None:
        import cv2

        self._cv2 = cv2
        self._min_confidence = min_confidence
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def process_image(self, image: Image) -> ImageDetections2D[Detection2DBBox]:
        import numpy as np

        scaled, scale = image.resize_to_fit(640, 480)
        frame = scaled.to_opencv()
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        if len(rects) == 0:
            return ImageDetections2D(image=image, detections=[])

        flat_weights = np.ravel(weights)
        detections: list[Detection2DBBox] = []
        for i, (x, y, width, height) in enumerate(rects):
            confidence = float(flat_weights[i]) if i < len(flat_weights) else 1.0
            if confidence < self._min_confidence:
                continue

            inv_scale = 1.0 / scale
            x1 = float(x * inv_scale)
            y1 = float(y * inv_scale)
            x2 = float((x + width) * inv_scale)
            y2 = float((y + height) * inv_scale)
            detection = Detection2DBBox(
                bbox=(x1, y1, x2, y2),
                track_id=i,
                class_id=0,
                confidence=confidence,
                name="person",
                ts=image.ts,
                image=image,
            )
            if detection.is_valid():
                detections.append(detection)

        return ImageDetections2D(image=image, detections=detections)

    def stop(self) -> None:
        pass


class EmbeddingPersonDeduper:
    """Deduplicates found people by comparing embeddings from person crops."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        similarity_threshold: float,
        padding_px: int,
        max_embeddings_per_person: int,
    ) -> None:
        self._model_factory = model_factory
        self._model: Any | None = None
        self._similarity_threshold = similarity_threshold
        self._padding_px = padding_px
        self._max_embeddings_per_person = max(1, max_embeddings_per_person)
        self._known_people: dict[int, list[np.ndarray]] = {}
        self._next_person_id = 0

    def register_found(self, detection: Detection2DBBox) -> tuple[bool, int, float | None]:
        embedding = self._embedding_for_detection(detection)
        best_person_id, best_similarity = self._best_match(embedding)

        if best_person_id is not None and best_similarity >= self._similarity_threshold:
            self._remember(best_person_id, embedding)
            return False, best_person_id, best_similarity

        person_id = self._next_person_id
        self._next_person_id += 1
        self._remember(person_id, embedding)
        return True, person_id, best_similarity

    def reset(self) -> None:
        self._known_people.clear()
        self._next_person_id = 0

    def stop(self) -> None:
        if self._model is not None and hasattr(self._model, "stop"):
            self._model.stop()
        self._model = None

    def _embedding_for_detection(self, detection: Detection2DBBox) -> np.ndarray:
        model = self._model_instance()
        embedding = model.embed(detection.cropped_image(padding=self._padding_px))
        assert not isinstance(embedding, list), "Expected one embedding for one person crop"
        embedding = embedding.to_cpu()
        vector = np.ravel(embedding.to_numpy()).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _model_instance(self) -> Any:
        if self._model is None:
            self._model = self._model_factory()
            if hasattr(self._model, "start"):
                self._model.start()
        return self._model

    def _best_match(self, embedding: np.ndarray) -> tuple[int | None, float | None]:
        best_person_id: int | None = None
        best_similarity: float | None = None

        for person_id, stored_embeddings in self._known_people.items():
            similarities = [float(embedding @ stored) for stored in stored_embeddings]
            if not similarities:
                continue

            similarity = max(similarities)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id

        return best_person_id, best_similarity

    def _remember(self, person_id: int, embedding: np.ndarray) -> None:
        embeddings = self._known_people.setdefault(person_id, [])
        embeddings.append(embedding)
        del embeddings[: max(0, len(embeddings) - self._max_embeddings_per_person)]


def _create_person_detector(
    backend: DetectorBackend,
    min_confidence: float,
) -> PersonDetector:
    if backend == "hog":
        logger.info("Using CPU OpenCV HOG for hide and seek person detection")
        return Cv2HogPersonDetector(min_confidence)

    if backend == "yolo-cpu":
        logger.info("Using CPU YOLO for hide and seek person detection")
        return YoloPersonDetectorAdapter("cpu")

    try:
        logger.info("Trying CPU YOLO for hide and seek person detection")
        return YoloPersonDetectorAdapter("cpu")
    except Exception:
        logger.warning(
            "Failed to initialize CPU YOLO detector; trying fallback",
            exc_info=True,
        )

    logger.warning("Falling back to CPU OpenCV HOG for hide and seek person detection")
    return Cv2HogPersonDetector(min_confidence)


def _create_person_deduper(
    backend: PersonDeduperBackend,
    similarity_threshold: float,
    padding_px: int,
    max_embeddings_per_person: int,
) -> PersonDeduper:
    if backend == "mobileclip":
        from dimos.models.embedding.mobileclip import MobileCLIPModel

        logger.info("Using MobileCLIP for hide and seek person dedupe")
        return EmbeddingPersonDeduper(
            model_factory=MobileCLIPModel,
            similarity_threshold=similarity_threshold,
            padding_px=padding_px,
            max_embeddings_per_person=max_embeddings_per_person,
        )

    from dimos.models.embedding.treid import TorchReIDModel

    logger.info("Using TorchReID for hide and seek person dedupe")
    return EmbeddingPersonDeduper(
        model_factory=TorchReIDModel,
        similarity_threshold=similarity_threshold,
        padding_px=padding_px,
        max_embeddings_per_person=max_embeddings_per_person,
    )


def _create_router(
    global_config: GlobalConfig,
    router_name: PatrolRouterName = "coverage",
) -> PatrolRouter:
    clearance_radius_m = global_config.robot_width * 0.5
    return create_patrol_router(router_name, clearance_radius_m)


def _countdown_text(number: int) -> str:
    if number < 0:
        return f"minus {_countdown_text(abs(number))}"
    if number < 10:
        return _COUNTDOWN_ONES[number]
    if number < 20:
        return _COUNTDOWN_TEENS[number - 10]
    if number < 100:
        tens, ones = divmod(number, 10)
        if ones == 0:
            return _COUNTDOWN_TENS[tens]
        return f"{_COUNTDOWN_TENS[tens]} {_COUNTDOWN_ONES[ones]}"
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        if remainder == 0:
            return f"{_COUNTDOWN_ONES[hundreds]} hundred"
        return f"{_COUNTDOWN_ONES[hundreds]} hundred {_countdown_text(remainder)}"
    return " ".join(_COUNTDOWN_ONES[int(digit)] for digit in str(number))


class HideAndSeekModule(Module):
    """Hide and seek game state machine for Unitree Go2."""

    config: HideAndSeekModuleConfig

    odom: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    goal_reached: In[Bool]
    color_image: In[Image]

    goal_request: Out[PoseStamped]
    cmd_vel: Out[Twist]
    detection: Out[Image]
    bounded_global_costmap: Out[OccupancyGrid]
    cached_global_costmap: Out[OccupancyGrid]
    hide_and_seek_state: Out[String]

    _planner_spec: ReplanningAStarPlannerSpec
    _speak_skill: SpeakSkillSpec
    _sport_skill: SportCommandSkillSpec
    _explorer_spec: FrontierExplorerSpec | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._router = _create_router(self.config.g)
        self._fallback_router = _create_router(self.config.g, "random")
        self._detector = _create_person_detector(
            self.config.detector_backend,
            self.config.detector_min_confidence,
        )
        self._person_deduper: PersonDeduper | None = None
        if self.config.enable_person_deduper:
            try:
                self._person_deduper = _create_person_deduper(
                    self.config.person_deduper_backend,
                    self.config.person_deduper_similarity_threshold,
                    self.config.person_deduper_padding_px,
                    self.config.person_deduper_max_embeddings_per_person,
                )
            except Exception:
                logger.warning("Failed to initialize hide and seek person deduper", exc_info=True)
                self.config.enable_person_deduper = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._scan_event = threading.Event()
        self._costmap_event = threading.Event()
        self._goal_reached_event = threading.Event()
        self._main_thread: threading.Thread | None = None
        self._scan_thread: threading.Thread | None = None
        self._state: State = "IDLE"
        self._latest_pose: PoseStamped | None = None
        self._latest_image: Image | None = None
        self._latest_costmap: OccupancyGrid | None = None
        self._mapping_origin_xy: tuple[float, float] | None = None
        self._has_active_goal = False
        self._detection_streak = 0
        self._target_people = 3
        self._found_people = 0
        self._awaiting_person_clear = False
        self._last_no_patrol_goal_log_time = 0.0
        self._last_map_cache_save_time = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_global_costmap)))
        self.register_disposable(Disposable(self.goal_reached.subscribe(self._on_goal_reached)))
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))

    @rpc
    def stop(self) -> None:
        self._stop_hide_and_seek_internal()
        self._detector.stop()
        if self._person_deduper is not None:
            self._person_deduper.stop()
        super().stop()

    @skill
    def start_hide_and_seek(
        self,
        countdown_seconds: int = 10,
        max_search_seconds: float = 180.0,
        target_people: int = 3,
    ) -> str:
        """Start a hide and seek game.

        The robot speaks a countdown, patrols the known room, looks for a person,
        says "Found you." for each person, and ends after finding the target count.

        Args:
            countdown_seconds: Seconds to count down before the robot starts searching.
            max_search_seconds: Maximum time to search before giving up.
            target_people: Number of people the robot must find before ending.
        """
        if countdown_seconds < 0:
            return "countdown_seconds must be non-negative."
        if max_search_seconds < 0:
            return "max_search_seconds must be non-negative."
        if target_people <= 0:
            return "target_people must be positive."

        with self._lock:
            if self._main_thread is not None and self._main_thread.is_alive():
                return "Hide and seek is already running. Use `stop_hide_and_seek` to stop."

            self._router.reset()
            self._fallback_router.reset()
            if self._person_deduper is not None:
                self._person_deduper.reset()
            self._configure_navigation_for_search()
            self._mapping_origin_xy = self._pose_xy(self._latest_pose)
            self._stop_event.clear()
            self._costmap_event.clear()
            self._goal_reached_event.clear()
            self._has_active_goal = False
            self._detection_streak = 0
            self._target_people = int(target_people)
            self._found_people = 0
            self._awaiting_person_clear = False
            self._main_thread = threading.Thread(
                target=self._main_loop,
                args=(int(countdown_seconds), float(max_search_seconds)),
                daemon=True,
                name=f"{self.__class__.__name__}-main",
            )
            self._main_thread.start()

        return (
            "Hide and seek started. The robot will count down, search the room, "
            f"and celebrate each time it finds one of {int(target_people)} people."
        )

    @skill
    def stop_hide_and_seek(self) -> str:
        """Stop the current hide and seek game."""
        self._stop_hide_and_seek_internal()
        return "Hide and seek stopped."

    @skill
    def hide_and_seek_status(self) -> str:
        """Return the current hide and seek state."""
        with self._lock:
            state = self._state
            found_people = self._found_people
            target_people = self._target_people
        return f"Hide and seek state: {state}. Found people: {found_people}/{target_people}."

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_pose = msg
            if self._mapping_origin_xy is None and self._state in (
                "MAPPING",
                "COUNTDOWN",
                "SEARCHING",
                "FOUND",
            ):
                self._mapping_origin_xy = self._pose_xy(msg)
        self._router.handle_odom(msg)
        self._fallback_router.handle_odom(msg)

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        bounded = self._limit_costmap_to_mapping_radius(msg)
        self.bounded_global_costmap.publish(bounded)
        with self._lock:
            self._latest_costmap = bounded
        self._router.handle_occupancy_grid(bounded)
        self._fallback_router.handle_occupancy_grid(bounded)
        self._costmap_event.set()
        self._save_costmap_cache_if_useful(bounded)

    def _on_goal_reached(self, _msg: Bool) -> None:
        self._goal_reached_event.set()

    def _on_color_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    def _main_loop(self, countdown_seconds: int, max_search_seconds: float) -> None:
        try:
            cached_costmap = self._load_cached_costmap()
            if cached_costmap is None and (
                self.config.initial_map_scan_seconds > 0 or self.config.use_frontier_mapping
            ):
                self._transition_to("MAPPING")
                self._run_mapping_phase()
                if self._stop_event.is_set():
                    return
            elif cached_costmap is not None:
                self._prime_cached_costmap(cached_costmap)

            self._configure_navigation_for_search()
            self._transition_to("COUNTDOWN")
            self._run_countdown(countdown_seconds)
            if self._stop_event.is_set():
                return

            self._transition_to("SEARCHING")
            self._speak_skill.speak("Ready or not, here I come!", blocking=False)

            deadline = time.monotonic() + max_search_seconds
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                self._search_step()

            if not self._stop_event.is_set() and not self._found_goal_complete():
                self._speak_skill.speak("I could not find everyone.", blocking=False)
        finally:
            self._stop_scanning_in_place()
            self.cmd_vel.publish(Twist.zero())
            self._restore_navigation_after_search()
            self._transition_to("IDLE")
            with self._lock:
                self._main_thread = None

    def _run_countdown(self, countdown_seconds: int) -> None:
        for remaining in range(countdown_seconds, 0, -1):
            if self._stop_event.is_set():
                return
            self._speak_skill.speak(_countdown_text(remaining), blocking=False)
            self._stop_event.wait(timeout=1.0)

    def _search_step(self) -> None:
        wait_after_step = False
        if not self._has_active_goal:
            goal = self._next_patrol_goal()
            if goal is None:
                self._start_scanning_in_place()
                wait_after_step = True
            else:
                self._stop_scanning_in_place()
                self._goal_reached_event.clear()
                self.goal_request.publish(goal)
                self._has_active_goal = True

        if self._goal_reached_event.is_set():
            self._goal_reached_event.clear()
            self._has_active_goal = False

        with self._lock:
            image = self._latest_image
        if image is None:
            timeout = self.config.search_retry_seconds if wait_after_step else 0.01
            self._stop_event.wait(timeout=timeout)
            return

        best = self._find_best_person(image)
        if best is None:
            self._detection_streak = 0
            self._awaiting_person_clear = False
            if wait_after_step:
                self._stop_event.wait(timeout=self.config.search_retry_seconds)
            return

        self.detection.publish(ImageDetections2D(image=image, detections=[best]).annotated_image())
        if self._awaiting_person_clear:
            self._detection_streak = 0
            if wait_after_step:
                self._stop_event.wait(timeout=self.config.search_retry_seconds)
            return

        self._detection_streak += 1
        if self._detection_streak >= self.config.confirmation_frames:
            self._handle_found(best)
        elif wait_after_step:
            self._stop_event.wait(timeout=self.config.search_retry_seconds)

    def _find_best_person(self, image: Image) -> Detection2DBBox | None:
        try:
            all_detections = self._detector.process_image(image)
        except Exception:
            logger.error("Hide and seek person detection failed", exc_info=True)
            return None

        persons = [
            detection
            for detection in all_detections.detections
            if detection.name == "person"
            and detection.is_valid()
            and detection.bbox_2d_volume() >= self.config.min_detection_area_px
        ]
        if not persons:
            return None

        return max(persons, key=lambda detection: detection.bbox_2d_volume())

    def _next_patrol_goal(self) -> PoseStamped | None:
        goal = self._router.next_goal()
        if goal is not None:
            return goal

        goal = self._fallback_router.next_goal()
        if goal is not None:
            logger.info("Coverage patrol has no goal; using random patrol fallback")
            return goal

        return None

    def _handle_found(self, detection: Detection2DBBox) -> None:
        if self._is_duplicate_found_person(detection):
            with self._lock:
                self._detection_streak = 0
                self._awaiting_person_clear = True
            return

        with self._lock:
            self._found_people += 1
            found_people = self._found_people
            target_people = self._target_people
            self._detection_streak = 0

        self._stop_scanning_in_place()
        self._cancel_current_goal()
        self._has_active_goal = False
        self.cmd_vel.publish(Twist.zero())
        self._transition_to("FOUND")
        self._announce_found()
        self._celebrate_found()
        if found_people >= target_people:
            self._stop_event.set()
            return

        logger.info(
            "hide and seek found one person, continuing search",
            found_people=found_people,
            target_people=target_people,
        )
        self._awaiting_person_clear = True
        self._transition_to("SEARCHING")

    def _is_duplicate_found_person(self, detection: Detection2DBBox) -> bool:
        if not self.config.enable_person_deduper or self._person_deduper is None:
            return False

        try:
            is_new, person_id, similarity = self._person_deduper.register_found(detection)
        except Exception:
            logger.warning("Hide and seek person dedupe failed", exc_info=True)
            return False

        if is_new:
            logger.info(
                "hide and seek registered new person identity",
                person_id=person_id,
                best_similarity=similarity,
            )
            return False

        logger.info(
            "hide and seek ignored duplicate person identity",
            person_id=person_id,
            similarity=similarity,
        )
        return True

    def _found_goal_complete(self) -> bool:
        with self._lock:
            return self._found_people >= self._target_people

    def _run_mapping_phase(self) -> None:
        if self.config.initial_map_scan_seconds > 0:
            self._speak_skill.speak("Scanning the room.", blocking=False)
            self._scan_for_map(self.config.initial_map_scan_seconds)
            if self._stop_event.is_set():
                return

        if not self.config.use_frontier_mapping:
            return

        explorer = getattr(self, "_explorer_spec", None)
        if explorer is None:
            logger.info("No frontier explorer available; hide and seek mapping used scan only")
            return

        self._configure_navigation_for_mapping()
        self._speak_skill.speak("Exploring the room.", blocking=False)

        try:
            started = explorer.explore()
        except Exception:
            logger.warning("Failed to start hide and seek frontier exploration", exc_info=True)
            return

        if not started:
            logger.info("Hide and seek frontier exploration was already active")

        timed_out = self._wait_for_frontier_exploration(explorer)
        if timed_out:
            logger.warning(
                "Hide and seek frontier exploration timed out",
                timeout_seconds=self.config.map_exploration_timeout_seconds,
            )
            try:
                explorer.stop_exploration()
            except Exception:
                logger.warning("Failed to stop timed-out frontier exploration", exc_info=True)

        with self._lock:
            latest_costmap = self._latest_costmap
        if latest_costmap is not None and self._costmap_is_useful(latest_costmap):
            self._prime_cached_costmap(latest_costmap)

    def _wait_for_frontier_exploration(self, explorer: FrontierExplorerSpec) -> bool:
        deadline = time.monotonic() + max(0.0, self.config.map_exploration_timeout_seconds)
        poll_seconds = max(0.0, self.config.map_exploration_poll_seconds)

        while not self._stop_event.is_set():
            try:
                active = explorer.is_exploration_active()
            except Exception:
                logger.warning("Failed to read frontier exploration state", exc_info=True)
                return False

            if not active:
                return False

            if time.monotonic() >= deadline:
                return True

            self._stop_event.wait(timeout=poll_seconds)

        try:
            if explorer.is_exploration_active():
                explorer.stop_exploration()
        except Exception:
            logger.warning("Failed to stop interrupted frontier exploration", exc_info=True)
        return False

    def _scan_for_map(self, scan_seconds: float) -> None:
        self._start_scanning_in_place()
        self._stop_event.wait(timeout=max(0.0, scan_seconds))
        self._stop_scanning_in_place()
        if self._stop_event.is_set():
            return

        self._router.reset()
        self._fallback_router.reset()
        self._costmap_event.clear()
        if not self._costmap_event.wait(timeout=self.config.map_refresh_wait_seconds):
            logger.warning(
                "No fresh costmap received after hide and seek map scan",
                timeout_seconds=self.config.map_refresh_wait_seconds,
            )
        with self._lock:
            latest_costmap = self._latest_costmap
        if latest_costmap is not None and self._costmap_is_useful(latest_costmap):
            self._prime_cached_costmap(latest_costmap)

    def _resolve_map_cache_path(self) -> Path | None:
        cache_path = self.config.map_cache_path
        if cache_path is None:
            return None

        path = Path(cache_path).expanduser()
        if not path.is_absolute():
            path = DIMOS_PROJECT_ROOT / path
        return path

    def _load_cached_costmap(self) -> OccupancyGrid | None:
        if not self.config.use_cached_map:
            return None

        path = self._resolve_map_cache_path()
        if path is None or not path.exists():
            return None

        try:
            costmap = OccupancyGrid.lcm_decode(path.read_bytes())
        except Exception:
            logger.warning(
                "Failed to load hide and seek costmap cache", path=str(path), exc_info=True
            )
            return None
        costmap = self._limit_costmap_to_mapping_radius(costmap)

        if not self._costmap_is_useful(costmap):
            logger.warning(
                "Ignoring hide and seek costmap cache with too little free space",
                path=str(path),
                free_cells=costmap.free_cells,
                min_free_cells=self.config.min_cached_costmap_free_cells,
            )
            return None

        logger.info(
            "Loaded hide and seek costmap cache",
            path=str(path),
            width=costmap.width,
            height=costmap.height,
            free_cells=costmap.free_cells,
        )
        return costmap

    def _prime_cached_costmap(self, costmap: OccupancyGrid) -> None:
        self._router.reset()
        self._fallback_router.reset()
        self._router.handle_occupancy_grid(costmap)
        self._fallback_router.handle_occupancy_grid(costmap)

        with self._lock:
            latest_pose = self._latest_pose
        if latest_pose is not None:
            self._router.handle_odom(latest_pose)
            self._fallback_router.handle_odom(latest_pose)

        self.cached_global_costmap.publish(costmap)
        self._costmap_event.set()

    def _save_costmap_cache_if_useful(self, costmap: OccupancyGrid) -> None:
        if not self.config.save_map_cache or not self._costmap_is_useful(costmap):
            return

        now = time.monotonic()
        if now - self._last_map_cache_save_time < self.config.map_cache_save_interval_seconds:
            return

        path = self._resolve_map_cache_path()
        if path is None:
            return

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(costmap.lcm_encode())
        except Exception:
            logger.warning(
                "Failed to save hide and seek costmap cache", path=str(path), exc_info=True
            )
            return

        self._last_map_cache_save_time = now
        logger.info(
            "Saved hide and seek costmap cache",
            path=str(path),
            width=costmap.width,
            height=costmap.height,
            free_cells=costmap.free_cells,
        )

    def _costmap_is_useful(self, costmap: OccupancyGrid) -> bool:
        return (
            costmap.width > 0
            and costmap.height > 0
            and (costmap.free_cells >= self.config.min_cached_costmap_free_cells)
        )

    def _limit_costmap_to_mapping_radius(self, costmap: OccupancyGrid) -> OccupancyGrid:
        radius = self.config.map_max_radius_m
        if radius is None or radius <= 0:
            return costmap

        with self._lock:
            origin_xy = self._mapping_origin_xy
        if origin_xy is None or costmap.grid.size == 0:
            return costmap

        center_x, center_y = origin_xy
        rows, cols = np.indices(costmap.grid.shape)
        world_x = costmap.origin.position.x + (cols + 0.5) * costmap.resolution
        world_y = costmap.origin.position.y + (rows + 0.5) * costmap.resolution
        outside_radius = ((world_x - center_x) ** 2 + (world_y - center_y) ** 2) > radius**2
        if not np.any(outside_radius):
            return costmap

        grid = costmap.grid.copy()
        grid[outside_radius] = CostValues.OCCUPIED
        bounded = OccupancyGrid(
            grid=grid,
            resolution=costmap.resolution,
            origin=costmap.origin,
            frame_id=costmap.frame_id,
            ts=costmap.ts,
        )
        bounded.info = costmap.info
        return bounded

    def _pose_xy(self, pose: PoseStamped | None) -> tuple[float, float] | None:
        if pose is None:
            return None
        return (pose.position.x, pose.position.y)

    def _start_scanning_in_place(self) -> None:
        now = time.monotonic()
        if now - self._last_no_patrol_goal_log_time >= self.config.no_patrol_goal_log_seconds:
            logger.info("No hide and seek patrol goal available, scanning in place")
            self._last_no_patrol_goal_log_time = now

        self._scan_event.set()
        with self._lock:
            thread = self._scan_thread
            if thread is not None and thread.is_alive():
                return

            self._scan_thread = threading.Thread(
                target=self._scan_loop,
                daemon=True,
                name=f"{self.__class__.__name__}-scan",
            )
            self._scan_thread.start()

    def _scan_loop(self) -> None:
        while self._scan_event.is_set() and not self._stop_event.is_set():
            self.cmd_vel.publish(
                Twist(
                    linear=[0.0, 0.0, 0.0],
                    angular=[0.0, 0.0, self.config.scan_angular_speed],
                )
            )
            self._scan_event.wait(timeout=self.config.scan_command_period_seconds)
        self.cmd_vel.publish(Twist.zero())

    def _stop_scanning_in_place(self) -> None:
        self._scan_event.clear()
        with self._lock:
            thread = self._scan_thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if not thread.is_alive():
                with self._lock:
                    self._scan_thread = None
        elif thread is None:
            self.cmd_vel.publish(Twist.zero())

    def _announce_found(self) -> None:
        speak_skill = getattr(self, "_speak_skill", None)
        if speak_skill is None:
            return

        try:
            speak_skill.speak("Found you.", blocking=False)
        except Exception:
            logger.warning("Failed to speak hide and seek found line", exc_info=True)

    def _celebrate_found(self) -> None:
        sport_skill = getattr(self, "_sport_skill", None)
        if sport_skill is None:
            return

        for command_name in ("BalanceStand", "Hello"):
            try:
                result = sport_skill.execute_sport_command(command_name)
                logger.info(
                    "hide and seek celebration command",
                    command=command_name,
                    result=result,
                )
            except Exception:
                logger.warning(
                    f"Failed to run hide and seek celebration command {command_name}",
                    exc_info=True,
                )

    def _configure_navigation_for_search(self) -> None:
        planner = getattr(self, "_planner_spec", None)
        if planner is None:
            return
        planner.set_replanning_enabled(False)
        planner.set_safe_goal_clearance(self.config.g.robot_rotation_diameter / 2 + EXTRA_CLEARANCE)

    def _configure_navigation_for_mapping(self) -> None:
        planner = getattr(self, "_planner_spec", None)
        if planner is None:
            return
        planner.set_replanning_enabled(True)
        planner.reset_safe_goal_clearance()

    def _restore_navigation_after_search(self) -> None:
        planner = getattr(self, "_planner_spec", None)
        if planner is not None:
            planner.set_replanning_enabled(True)
            planner.reset_safe_goal_clearance()
        self._cancel_current_goal()

    def _cancel_current_goal(self) -> None:
        planner = getattr(self, "_planner_spec", None)
        if planner is not None:
            planner.cancel_goal()

    def _transition_to(self, new_state: State) -> None:
        with self._lock:
            old_state = self._state
            self._state = new_state
        logger.info("hide and seek state transition", old=old_state, new=new_state)
        self.hide_and_seek_state.publish(String(new_state))

    def _stop_hide_and_seek_internal(self) -> None:
        self._stop_event.set()
        self._stop_scanning_in_place()
        self._restore_navigation_after_search()
        self.cmd_vel.publish(Twist.zero())

        with self._lock:
            thread = self._main_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if not thread.is_alive():
                with self._lock:
                    self._main_thread = None
