"""
External image generator — stateless functions for communicating with a
diffusers-ui server to offload sample generation.
"""
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def build_board_name(project_name: str, log_time: str, global_step: int, suffix: str = "") -> str:
    """
    Build a board name string: <project_name>-<log_time>-gs<global_step>[-<suffix>]
    """
    base = f"{project_name}-{log_time}-gs{global_step}"
    if suffix:
        base += f"-{suffix}"
    return base


def create_board(base_url: str, board_name: str) -> str:
    """
    POST /api/v1/boards?board_name=<board_name>
    Returns the board_id UUID string.
    """
    resp = requests.post(f"{base_url}/api/v1/boards", params={"board_name": board_name})
    resp.raise_for_status()
    data = resp.json()
    board_id = data.get("id") or data.get("board_id") or data.get("board", {}).get("id")
    if not board_id:
        raise RuntimeError(f"Could not extract board_id from response: {data}")
    return board_id


def _pipeline_base(pipeline) -> str:
    """Determine the 'base' string for model installation based on pipeline type."""
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, SanaPipeline
    if isinstance(pipeline, StableDiffusionPipeline):
        return "sd-1"
    if isinstance(pipeline, StableDiffusionXLPipeline):
        return "sdxl"
    if isinstance(pipeline, SanaPipeline):
        return "sana"
    # fallback: try to infer from class name
    name = type(pipeline).__name__
    if "SDXL" in name or "StableDiffusionXl" in name.lower():
        return "sdxl"
    if "Sana" in name:
        return "sana"
    return "sd-1"


def install_model(base_url: str, model_path: str, model_name: str, pipeline) -> str:
    """
    1. Save pipeline to model_path via save_pretrained.
    2. Determine base from pipeline type.
    3. POST /api/v2/models/install?source=<model_path> to register.
    Returns model_key from response.
    """
    pipeline.save_pretrained(model_path)
    base = _pipeline_base(pipeline)

    body = {
        "name": model_name,
        "base": base,
        "source_type": "local_path",
    }
    resp = requests.post(
        f"{base_url}/api/v2/models/install",
        params={"source": model_path},
        json=body,
    )
    resp.raise_for_status()
    data = resp.json()
    model_key = data.get("model_key") or data.get("key") or data.get("id")
    if not model_key:
        raise RuntimeError(f"Could not extract model_key from response: {data}")
    return model_key


def unload_model(base_url: str) -> None:
    """
    POST /api/v2/models/unload_cache
    Unloads the model pipeline from the server's GPU.
    May return 409 if a generation is still in progress — caller should
    only call this after polling completes.
    """
    try:
        resp = requests.post(f"{base_url}/api/v2/models/unload_cache")
        if resp.status_code == 409:
            logger.warning("Server returned 409 — generation still in progress?")
        else:
            resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to unload model cache: {e}")


def uninstall_model(base_url: str, model_key: str) -> None:
    """
    DELETE /api/v2/models/{model_key}
    """
    resp = requests.delete(f"{base_url}/api/v2/models/{model_key}")
    resp.raise_for_status()


def _field_mapping() -> dict:
    """Mapping from external dict field names to diffusers-ui queue field names."""
    return {
        "positive_prompt": "prompt",
        "negative_prompt": "negative_prompt",
        "seed": "seed",
        "width": "width",
        "height": "height",
        "steps": "steps",
        "cfg_scale": "cfg",
        "cfg_rescale_multiplier": "cfg_rescale_multiplier",
        "scheduler": "sampler",
    }


def enqueue_batch(base_url: str, items: list[dict], board_id: str, model_key: str) -> list[int]:
    """
    For each item dict, map fields to the queue native format and enqueue.
    POST /api/v1/queue/default/enqueue_batch
    Returns a list of item_ids.
    """
    field_map = _field_mapping()
    queue_items = []
    for item in items:
        mapped = {}
        for ext_key, queue_key in field_map.items():
            if ext_key in item:
                mapped[queue_key] = item[ext_key]
            elif queue_key in item:
                mapped[queue_key] = item[queue_key]
        mapped["model"] = model_key
        mapped["board_id"] = board_id
        mapped["count"] = 1
        queue_items.append(mapped)

    body = {
        "name": "enqueue_batch",
        "items": queue_items,
    }
    resp = requests.post(f"{base_url}/api/v1/queue/default/enqueue_batch", json=body)
    resp.raise_for_status()
    data = resp.json()
    item_ids = []
    if isinstance(data, list):
        for entry in data:
            item_ids.append(entry.get("item_id") or entry.get("id") or entry)
    elif isinstance(data, dict):
        items_list = data.get("items", [])
        for entry in items_list:
            item_ids.append(entry.get("item_id") or entry.get("id"))
    return item_ids


def fetch_status(base_url: str, item_id: int) -> str:
    """
    GET /api/v1/queue/default/i/{item_id}
    Returns the status string (e.g. "completed", "failed", "in_progress").
    """
    resp = requests.get(f"{base_url}/api/v1/queue/default/i/{item_id}")
    resp.raise_for_status()
    data = resp.json()
    return data.get("status", "unknown")


def fetch_image_name(base_url: str, item_id: int) -> str:
    """
    GET /api/v1/queue/default/i/{item_id}
    Returns the image_name for a completed item.
    """
    resp = requests.get(f"{base_url}/api/v1/queue/default/i/{item_id}")
    resp.raise_for_status()
    data = resp.json()
    image_name = data.get("image_name")
    if not image_name:
        # Some servers return it nested
        for result in data.get("results", []):
            image_name = result.get("image_name") or result.get("image")
            if image_name:
                break
    if not image_name:
        raise RuntimeError(f"Could not find image_name for item {item_id}: {data}")
    return image_name


def poll_until_done(
    base_url: str,
    item_ids: list[int],
    idle_timeout: float = 60.0,
    poll_interval: float = 2.0,
) -> dict[int, str]:
    """
    Poll all item_ids until they complete, fail, or idle_timeout is reached.
    Returns dict of {item_id: image_name} for completed items.
    """
    last_status_change = time.monotonic()
    statuses = {item_id: "pending" for item_id in item_ids}
    completed = {}

    while True:
        any_changed = False
        for item_id in item_ids:
            try:
                current = fetch_status(base_url, item_id)
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch status for item {item_id}: {e}")
                continue
            if current != statuses[item_id]:
                any_changed = True
                statuses[item_id] = current

        if any_changed:
            last_status_change = time.monotonic()

        for item_id in item_ids:
            if statuses[item_id] == "completed" and item_id not in completed:
                try:
                    image_name = fetch_image_name(base_url, item_id)
                    completed[item_id] = image_name
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch image name for item {item_id}: {e}")

        if all(s == "completed" for s in statuses.values()):
            break

        failed = [i for i in item_ids if statuses[i] == "failed"]
        if failed:
            logger.warning(f"Items failed: {failed}")
            remaining = [i for i in item_ids if i not in completed and statuses[i] != "failed"]
            if not remaining:
                break

        if time.monotonic() - last_status_change > idle_timeout:
            raise TimeoutError(
                f"No sample generation progress for {idle_timeout}s. "
                f"Completed: {list(completed.keys())}, "
                f"Statuses: {statuses}"
            )

        time.sleep(poll_interval)

    return completed


def download_image(base_url: str, image_name: str, dest_dir: str) -> Path:
    """
    GET /api/v1/images/i/{image_name}/full
    Save to dest_dir and return the local path.
    """
    dest = Path(dest_dir) / image_name
    resp = requests.get(f"{base_url}/api/v1/images/i/{image_name}/full")
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
