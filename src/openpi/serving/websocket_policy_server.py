import asyncio
import http
import logging
import os
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        # J20 (waist yaw) phase 3 detection state
        j20_was_high = False
        j20_low_count = 0
        j20_phase3_done = False
        j20_forced_last_step = False
        phase3_replay_idx = 0
        # Thresholds
        J20_HIGH_THRESH = 1.0
        J20_LOW_MIN = 0.05
        J20_LOW_MAX = 0.15
        J20_LOW_STEPS_NEEDED = 8
        # Load fixed phase 3 actions from dataset
        phase3_fixed = np.load(
            os.path.join(os.path.dirname(__file__), "phase3_fixed_actions.npy")
        )
        print(f"[J20] Loaded fixed phase 3 actions: {phase3_fixed.shape[0]} frames, {phase3_fixed.shape[1]} dims")
        # clean_the_desktop: multi-phase task with fixed action clips
        # Phase 0: auto(pen grab) → [108,112] → 1
        # Phase 1: auto(laptop cover) → [89,95] → 2
        # Phase 2: tissue clip → done → 3
        # Phase 3: auto(mouse grab) → [100,105] → 4
        # Phase 4: mouse clip → done → 5
        # Phase 5: auto(pen holder grab) → [96,99] → 6
        # Phase 6: pen holder clip → done → 7
        cd_phase = 0
        cd_first_action = None
        cd_clip_idx = 0
        cd_confirm_count = 0  # Require 2 consecutive detections before phase transition
        cd_last_trigger_phase = -1
        cd_step = 0           # Episode timestep counter (max 1830)
        cd_tissue_clip = np.load(os.path.join(os.path.dirname(__file__), "cd_tissue_handover.npy"))
        cd_mouse_clip = np.load(os.path.join(os.path.dirname(__file__), "cd_mouse_handover.npy"))
        cd_penholder_clip = np.load(os.path.join(os.path.dirname(__file__), "cd_penholder_placing.npy"))
        print(f"[Desktop] Loaded clips: tissue={cd_tissue_clip.shape[0]}, mouse={cd_mouse_clip.shape[0]}, penholder={cd_penholder_clip.shape[0]}")
        pb_phase = 0  # place_block_into_box phase: 0=watch for grab, 1=watch for release, 2=take first 5 steps, then done
        pb_initial_state = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                # Apply task-specific action post-processing
                task_name = obs.get("task_name", "")
                task_name_lower = task_name.lower()
                if task_name == "clean_the_desktop" or "desktop" in task_name_lower:
                    if "actions" in action:
                        raw_state = obs.get("state", None)
                        if raw_state is not None:
                            left_state = float(raw_state[14])
                            right_state = float(raw_state[15])

                            # Record first action frame for reset
                            if cd_first_action is None:
                                cd_first_action = action["actions"][0].copy()

                            # Pen grab failure: reset to initial pose
                            if left_state > 115:
                                action["actions"][:] = cd_first_action
                                print(f"[Desktop] Pen grab failed (left={left_state:.1f}), resetting")
                            if cd_phase == 0:
                                action["actions"][:, 1] -= 0.015
                            # if cd_phase == 5:
                            #     action["actions"][:, 8] -= 0.015
    
                            # Phase transitions: require 2 consecutive detections
                            # 0:auto(pen) 1:auto(laptop) 2:tissue_clip 3:auto(mouse)
                            # 4:mouse_clip 5:auto(penholder) 6:penholder_clip 7:auto(done)
                            trigger = None
                            if cd_phase == 0 and 108 <= left_state <= 112:
                                trigger = (1, 0, 0, f"pen grabbed (left={left_state:.1f})")
                            elif cd_phase == 1 and 89 <= left_state <= 107:
                                trigger = (2, 1, 0, f"tissue grabbed (left={left_state:.1f})")
                            elif cd_phase == 3 and 78 <= left_state <= 91:
                                trigger = (4, 3, 0, f"mouse grabbed (left={left_state:.1f})")
                            elif cd_phase == 5 and 92 <= right_state <= 103:
                                trigger = (6, 5, 0, f"pen holder grabbed (right={right_state:.1f})")

                            if trigger is not None:
                                new_phase, from_phase, init_clip_idx, msg = trigger
                                if cd_last_trigger_phase == from_phase:
                                    cd_confirm_count += 1
                                else:
                                    cd_last_trigger_phase = from_phase
                                    cd_confirm_count = 1
                                if cd_confirm_count >= 2:
                                    cd_phase = new_phase
                                    cd_clip_idx = init_clip_idx
                                    cd_confirm_count = 0
                                    cd_last_trigger_phase = -1
                                    print(f"[Desktop] Phase {from_phase}→{new_phase}: {msg}")
                            else:
                                cd_confirm_count = 0

                            # Apply fixed action clips for clip phases (2,4,6)
                            if cd_phase in (2, 4, 6):
                                if cd_clip_idx == -1:
                                    cd_clip_idx = 0
                                else:
                                    clip = {2: cd_tissue_clip, 4: cd_mouse_clip, 6: cd_penholder_clip}[cd_phase]
                                    chunk_size = action["actions"].shape[0]
                                    total = clip.shape[0]
                                    start = cd_clip_idx
                                    if start >= total:
                                        cd_phase += 1
                                        print(f"[Desktop] Clip done, phase {cd_phase-1}→{cd_phase} (auto)")
                                    else:
                                        end = min(start + chunk_size, total)
                                        fixed_chunk = clip[start:end]
                                        replace_dims = list(range(16))
                                        action["actions"][:fixed_chunk.shape[0], replace_dims] = fixed_chunk[:, replace_dims]
                                        cd_clip_idx = end
                                        print(f"[Desktop] Phase {cd_phase} clip [{start}:{end}/{total}]")

                            print(f"[Desktop] left={left_state:.1f} right={right_state:.1f} phase={cd_phase}")

                        action["actions"] = action["actions"][::3]
                        # Episode timestep: each frame in the chunk = 1 sim step
                        cd_step += action["actions"].shape[0]
                        print(f"[Desktop] cd_step={cd_step}")
                        if cd_step >= 1830:
                            cd_step = 0
                            cd_phase = 0
                            cd_clip_idx = 0
                            cd_confirm_count = 0
                            cd_last_trigger_phase = -1
                            print(f"[Desktop] Episode done (step >= 1830), reset to phase 0")
                elif task_name == "sorting_packages_continuous" or "continuous" in task_name_lower:
                    if "actions" in action:
                        raw_state = obs.get("state", None)
                        if raw_state is not None:
                            j20_state = float(raw_state[20])

                            # Detect high position (new cycle / reset)
                            if j20_state > J20_HIGH_THRESH:
                                j20_was_high = True
                                j20_low_count = 0
                                j20_phase3_done = False
                                j20_forced_last_step = False
                                phase3_replay_idx = 0

                            # Detect natural phase 3: state in low range AND not forced
                            if j20_was_high and J20_LOW_MIN <= j20_state <= J20_LOW_MAX and not j20_forced_last_step:
                                j20_low_count += 1
                                if j20_low_count >= J20_LOW_STEPS_NEEDED:
                                    j20_phase3_done = True

                            # Replay fixed phase 3 actions if current cycle skips phase 3
                            was_forced_this_step = False
                            if j20_was_high and not j20_phase3_done:
                                action_j20_all = action["actions"][:, 20]
                                if any(action_j20_all < J20_LOW_MIN):
                                    chunk_size = action["actions"].shape[0]
                                    total = phase3_fixed.shape[0]
                                    start = phase3_replay_idx % total
                                    end = start + chunk_size
                                    if end <= total:
                                        fixed_chunk = phase3_fixed[start:end].copy()
                                    else:
                                        fixed_chunk = np.concatenate([
                                            phase3_fixed[start:],
                                            phase3_fixed[:end - total]
                                        ])
                                    # Replace dims 0-15 (arms+grippers) and 20 (J20) from fixed
                                    # Keep dims 16-19 (frozen waist) from current model output
                                    replace_dims = list(range(16)) + [20]
                                    action["actions"][:fixed_chunk.shape[0], replace_dims] = fixed_chunk[:, replace_dims]
                                    phase3_replay_idx = end % total
                                    was_forced_this_step = True
                                    print(f"[J20] Replaying phase 3, frames [{start}:{end}] "
                                          f"(cycle buffer {total} frames)")

                            j20_forced_last_step = was_forced_this_step

                            print(f"[J20] state={j20_state:.4f} was_high={j20_was_high} "
                                  f"low_count={j20_low_count} phase3_done={j20_phase3_done} "
                                  f"forced={was_forced_this_step}")

                        action["actions"] = action["actions"][::2]
                elif task_name == "stock_and_straighten_shelf" or "straighten" in task_name_lower:
                    if "actions" in action:
                        action["actions"] = action["actions"][::2]
                elif task_name == "place_block_into_box" or "block" in task_name_lower:
                    if "actions" in action:
                        raw_state = obs.get("state", None)
                        if raw_state is not None:
                            # Record initial state for reset detection
                            if pb_initial_state is None:
                                pb_initial_state = raw_state.copy()

                            left_state = float(raw_state[14])

                            if pb_phase == 0:
                                # Phase 0: detect left gripper closes > 90 deg
                                if left_state > 90:
                                    pb_phase = 1
                                    print(f"[Block] Grab detected (left={left_state:.1f}), phase 0->1")

                            if pb_phase == 1:
                                # Phase 1: check if action chunk contains gripper open
                                left_grip_actions = action["actions"][:, 14]
                                if (left_grip_actions < 0.4).any():
                                    # Delay release: force gripper to stay closed for 5 frames
                                    open_mask = left_grip_actions < 0.4
                                    delay_frames = min(5, open_mask.sum())
                                    action["actions"][open_mask, 14] = 1.0  # keep gripper closed
                                    pb_phase = 2
                                    print(f"[Block] Delaying gripper open by {delay_frames} frames, phase 1->2")

                            if pb_phase == 2:
                                # Phase 2: detect return to initial pose
                                state_diff = np.linalg.norm(raw_state - pb_initial_state)
                                if state_diff < 1.0:
                                    pb_phase = 0
                                    pb_initial_state = None
                                    print(f"[Block] Returned to initial pose (diff={state_diff:.2f}), reset to phase 0")
                       
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
