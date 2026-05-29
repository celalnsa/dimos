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

import platform

from dimos.robot.unitree.go2.rerun_config import go2_rerun_pubsubs
from dimos.visualization.rerun.bridge import TopicTransportPubSub


def test_go2_rerun_pubsubs_include_pshm_camera_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    pubsubs = go2_rerun_pubsubs()
    adapters = [pubsub for pubsub in pubsubs if isinstance(pubsub, TopicTransportPubSub)]
    topic_names = {
        str(topic).split("#", 1)[0]
        for adapter in adapters
        for topic, _transport in adapter.topic_transports
    }

    assert "/color_image" in topic_names
