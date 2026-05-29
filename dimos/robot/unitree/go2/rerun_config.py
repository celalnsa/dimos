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
from typing import Any

from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.transport import pSHMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.visualization.rerun.bridge import TopicTransportPubSub


def go2_rerun_pubsubs() -> list[Any]:
    """Return pubsubs needed by Go2 Rerun viewers on the current platform."""
    pubsubs: list[Any] = [LCM()]
    if platform.system() != "Linux":
        pubsubs.append(
            TopicTransportPubSub(
                [
                    (
                        Topic("/color_image", Image),
                        pSHMTransport(
                            "color_image",
                            default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE,
                        ),
                    )
                ]
            )
        )
    return pubsubs
