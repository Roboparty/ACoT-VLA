import asyncio
import http
import logging
import time
import traceback
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from openpi.shared import task_stage

logger = logging.getLogger(__name__)


def _to_msgpack_compatible(obj):
    """Recursively cast unsupported numpy dtypes (e.g. bfloat16) before msgpack."""
    if isinstance(obj, np.ndarray):
        if str(obj.dtype) == "bfloat16":
            return obj.astype(np.float32)
        return obj
    if isinstance(obj, np.generic):
        if str(obj.dtype) == "bfloat16":
            return np.float32(obj)
        return obj
    if isinstance(obj, dict):
        return {k: _to_msgpack_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_msgpack_compatible(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_msgpack_compatible(v) for v in obj)
    return obj


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

        # Per-connection stage tracker state.
        current_task_name = None
        current_task_id = 0
        current_stage = 0
        history: list[int] = []
        history_len = 3
        votes_to_promote = 2
        max_stage = 14

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                task_name = (
                    obs.get("sub_task_name")
                    or obs.get("task_name")
                    or obs.get("task")
                    or obs.get("prompt")
                )
                if task_name is not None and hasattr(task_name, "item"):
                    task_name = task_name.item()
                if isinstance(task_name, bytes):
                    task_name = task_name.decode("utf-8")
                if task_name is not None:
                    task_name = str(task_name)

                if task_name != current_task_name:
                    current_task_name = task_name
                    current_task_id = task_stage.task_name_to_id(task_name, default_id=0)
                    current_stage = 0
                    history.clear()

                # Inject task/stage prompt tokens so client protocol remains unchanged.
                obs["tokenized_prompt"] = np.asarray([current_task_id, current_stage], dtype=np.int32)
                obs["tokenized_prompt_mask"] = np.asarray([True, True], dtype=bool)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Voting-based tracker update from model stage logits.
                if isinstance(action, dict) and "subtask_logits" in action:
                    logits = np.asarray(action["subtask_logits"])
                    if logits.ndim > 1:
                        logits = logits.reshape(-1)

                    # Use model output width as the stage bound when available.
                    model_max_stage = max(0, int(logits.shape[0]) - 1)

                    predicted_stage = int(np.argmax(logits))
                    predicted_stage = max(0, min(model_max_stage, predicted_stage))
                    history.append(predicted_stage)
                    if len(history) > history_len:
                        history.pop(0)

                    if len(history) == history_len:
                        next_stage = min(current_stage + 1, model_max_stage)
                        votes_for_next = sum(1 for p in history if p == next_stage)
                        votes_to_skip = sum(1 for p in history if p == min(next_stage + 1, model_max_stage))
                        votes_to_go_back = sum(1 for p in history if p == max(current_stage - 1, 0))

                        if votes_for_next >= votes_to_promote and next_stage > current_stage:
                            current_stage = next_stage
                            history.clear()
                        elif votes_to_skip == history_len and next_stage > current_stage:
                            current_stage = next_stage
                            history.clear()
                        elif votes_to_go_back == history_len and current_stage > 0:
                            current_stage -= 1
                            history.clear()

                if isinstance(action, dict):
                    action["stage_tracker"] = {
                        "task_id": int(current_task_id),
                        "stage": int(current_stage),
                    }

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                action = _to_msgpack_compatible(action)
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
