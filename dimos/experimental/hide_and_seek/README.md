# Hide and Seek

Experimental Unitree Go2 hide-and-seek game.

## Blueprint

```bash
dimos run unitree-go2-hns --robot-ip <robot-ip>
dimos --viewer rerun run unitree-go2-hns --robot-ip <robot-ip>
```

The blueprint exposes `start_hide_and_seek`, `stop_hide_and_seek`, and
`hide_and_seek_status` through MCP. With `--viewer rerun`, the Rerun layout shows
the raw camera, the annotated person-detection frame, and the 3D map.
On macOS, the Go2 camera stream uses shared memory instead of high-bandwidth
LCM; the HNS viewer config subscribes to that pSHM camera topic directly.

The HNS blueprint builds its own Go2 spatial stack and intentionally does not
include `SecurityModule`.

## Behavior

`start_hide_and_seek` runs a single internal state machine:

1. Load a cached navigation costmap from `.cache/hide_and_seek/global_costmap.lcm`.
2. If no usable cache exists, scan in place to seed an initial costmap.
3. Run frontier exploration to drive toward boundaries between known and
   unknown space until exploration completes or times out.
4. Speak a countdown with English words.
5. Patrol the known area with the coverage patrol router.
6. Fall back to a random safe patrol goal when coverage cannot pick a goal.
7. Detect a person in the Go2 camera stream for three consecutive frames.
8. Stop navigation, say "Found you.", enter `BalanceStand`, and run the Go2
   `Hello` greeting motion as a one-paw celebration.
9. Continue searching until three people have been found.

If the map is not ready and no patrol goal is available yet, the robot scans in
place while continuing to run person detection.
The in-place scan publishes velocity commands continuously, because Go2 WebRTC
stops a one-shot velocity command after a short timeout.
After each found person, the next count is blocked until the camera first sees
no person, which prevents one visible person from being counted repeatedly.
Set `ENABLE_PERSON_DEDUPER=1` to also deduplicate people who leave the frame and
come back later. When enabled, HNS compares the found person's cropped visual
embedding against people already found in the current game before incrementing
the found count. The default backend is TorchReID; if the optional model
dependency cannot initialize, HNS logs a warning and continues with dedupe
disabled.

The cached map is an `OccupancyGrid` costmap used by patrol routing and
navigation planning. It is separate from `SpatialMemory`, which stores visual
and semantic memory for perception queries rather than the planner's grid map.
Live costmaps are saved periodically when they contain enough free cells; on the
next game start, a usable cache skips the initial mapping scan and is republished
as `global_costmap` for the navigation stack.

The mapping phase reuses `WavefrontFrontierExplorer` from the Go2 smart
navigation stack. It does not try to walk every possible path. Instead, it picks
frontier goals with high expected information gain, navigates there, waits for
the costmap to expand, and repeats until no useful frontier remains or
`map_exploration_timeout_seconds` is reached.

Mapping is limited to `map_max_radius_m`, which defaults to 10 meters from the
robot pose when `start_hide_and_seek` is called. Cells outside that radius are
published as occupied in the HNS blueprint's bounded costmap, so frontier
exploration, planning, caching, and patrol all stay inside the demo area.

The detector defaults to `auto`, which starts with CPU YOLO and falls back to
HOG. The HNS module does not expose an MPS backend because Apple Metal compiler
failures can abort the worker process below Python.

1. YOLO person detection on CPU.
2. OpenCV HOG person detection as a CPU-only fallback.

HOG is only a last-resort fallback and is much less robust than YOLO.

## Verification

```bash
.venv/bin/python -m pytest dimos/experimental/hide_and_seek/test_hide_and_seek_module.py
.venv/bin/python -m pytest dimos/robot/unitree/go2/blueprints/agentic/test_unitree_go2_hns.py
CI=1 .venv/bin/python -m pytest dimos/robot/test_all_blueprints_generation.py
```
