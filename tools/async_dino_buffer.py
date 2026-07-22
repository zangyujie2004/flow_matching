import threading
import time
from collections import deque

import numpy as np
import torch
from torch import nn


class AsyncDinoBuffer:
    def __init__(
        self,
        dino_model: nn.Module,
        device: str = "cuda",
        sample_interval_frames: int = 4,
        deadline_ms: float = 132.0,
        store_local_features: bool = False,
    ):
        self.dino_model = dino_model.to(device)
        self.dino_model.eval()
        self.device = torch.device(device)
        self.sample_interval_frames = sample_interval_frames
        self.deadline_ms = deadline_ms
        self.store_local_features = store_local_features
        self.buffer = deque(maxlen=16)
        self.pending_frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.worker = None
        self.forward_1_times = []
        self.forward_2_times = []
        self.forward_3_times = []
        self.gpu_total_times = []
        self.stage_wall_times = []
        self.end_to_end_times = []
        self.patch_extract_times = []
        self.pooling_times = []
        self.view_stack_times = []
        self.buffer_append_times = []
        self.sample_total_times = []
        self.deadline_miss_count = 0
        self.processed_count = 0
        self.dropped_count = 0

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.new_frame_event.clear()
        self.dino_model.eval()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def stop(self):
        if self.worker is None:
            return
        self.stop_event.set()
        self.new_frame_event.set()
        self.worker.join()
        self.worker = None

    def submit_frame(self, frame_id, image_1, image_2, image_3=None, capture_time=None):
        if frame_id % self.sample_interval_frames != 0:
            return False
        images = [image_1, image_2]
        if image_3 is not None:
            images.append(image_3)
        if capture_time is None:
            capture_time = time.perf_counter()
        frame = (frame_id, images, capture_time)
        with self.lock:
            if self.stop_event.is_set():
                return False
            if self.pending_frame is not None:
                self.dropped_count += 1
            self.pending_frame = frame
            self.new_frame_event.set()
        return True

    def get_buffer(self):
        with self.lock:
            return list(self.buffer)

    def get_latest(self):
        with self.lock:
            if not self.buffer:
                return None
            return dict(self.buffer[-1])

    def get_feature_window(self):
        """Compatibility alias: return global features as (B,T,V,C)."""
        return self.get_global_feature_window()

    def get_global_feature_window(self):
        """Return detached patch-average features as (B,T=16,V,C), or None."""
        with self.lock:
            if len(self.buffer) < self.buffer.maxlen:
                return None
            features = [entry["global_feature"] for entry in self.buffer]

        first_shape = features[0].shape
        if any(feature.shape != first_shape for feature in features):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(f"DINO buffer contains inconsistent feature shapes: {shapes}")
        # Each item is (B,V,C); stacking time at dim=1 gives (B,T,V,C).
        window = torch.stack(features, dim=1).detach()
        if window.requires_grad or window.grad_fn is not None:
            raise RuntimeError("DINO feature window must not keep an autograd graph")
        return window

    def get_local_feature_window(self):
        """Return detached patch features as (B,T=16,V,N,C), or None."""
        with self.lock:
            if len(self.buffer) < self.buffer.maxlen:
                return None
            features = [entry["local_feature"] for entry in self.buffer]
        if any(feature is None for feature in features):
            return None
        window = torch.stack(features, dim=1).detach()  # (B,T,V,N,C)
        if window.requires_grad or window.grad_fn is not None:
            raise RuntimeError("DINO local feature window must not keep an autograd graph")
        return window

    def clear(self):
        """Release all feature tensors currently owned by the Buffer."""
        with self.lock:
            self.buffer.clear()

    def get_stats(self):
        with self.lock:
            timing_lists = {
                "backbone_forward_ms": list(
                    self.forward_1_times + self.forward_2_times + self.forward_3_times
                ),
                "forward_1_ms": list(self.forward_1_times),
                "forward_2_ms": list(self.forward_2_times),
                "forward_3_ms": list(self.forward_3_times),
                "gpu_total_ms": list(self.gpu_total_times),
                "stage_wall_ms": list(self.stage_wall_times),
                "end_to_end_ms": list(self.end_to_end_times),
                "patch_extract_ms": list(self.patch_extract_times),
                "pooling_ms": list(self.pooling_times),
                "view_stack_ms": list(self.view_stack_times),
                "buffer_append_ms": list(self.buffer_append_times),
                "sample_total_ms": list(self.sample_total_times),
            }
            counts = {
                "deadline_miss_count": self.deadline_miss_count,
                "processed_count": self.processed_count,
                "dropped_count": self.dropped_count,
                "buffer_length": len(self.buffer),
            }

        stats = {}
        for name, values in timing_lists.items():
            stats[name] = {
                "mean": float(np.mean(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
        stats.update(counts)
        return stats

    def _worker_loop(self):
        while True:
            if not self.stop_event.is_set():
                self.new_frame_event.wait(timeout=0.1)
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
                self.new_frame_event.clear()
            if frame is None and self.stop_event.is_set():
                break
            if frame is None:
                continue
            frame_id, images, capture_time = frame
            sample_start = time.perf_counter()
            global_feature, local_feature, timing = self._run_dino(images)
            ready_time = time.perf_counter()
            entry = {
                "frame_id": frame_id,
                # Old callers use "feature"; it now means patch-average global feature.
                "feature": global_feature,
                "global_feature": global_feature,
                "local_feature": local_feature,
                "capture_time": capture_time,
                "ready_time": ready_time,
                "stage_wall_ms": timing["stage_wall_ms"],
                "end_to_end_ms": (ready_time - capture_time) * 1000,
            }
            append_start = time.perf_counter()
            with self.lock:
                self.buffer.append(entry)
                append_ms = (time.perf_counter() - append_start) * 1000
                forward_times = timing["forward_times"]
                self.forward_1_times.append(forward_times[0])
                self.forward_2_times.append(forward_times[1])
                if len(forward_times) == 3:
                    self.forward_3_times.append(forward_times[2])
                self.gpu_total_times.append(sum(forward_times))
                self.patch_extract_times.append(timing["patch_extract_ms"])
                self.pooling_times.append(timing["pooling_ms"])
                self.view_stack_times.append(timing["view_stack_ms"])
                self.buffer_append_times.append(append_ms)
                self.sample_total_times.append((time.perf_counter() - sample_start) * 1000)
                self.stage_wall_times.append(timing["stage_wall_ms"])
                self.end_to_end_times.append(entry["end_to_end_ms"])
                self.processed_count += 1
                self.deadline_miss_count += timing["stage_wall_ms"] > self.deadline_ms

    def _run_dino(self, images):
        stage_start = time.perf_counter()
        images = [image.to(self.device) for image in images]
        local_features = []
        global_features = []
        forward_events = []
        pooling_events = []
        forward_times = []
        pooling_times = []
        patch_extract_ms = 0.0
        has_patch_api = hasattr(self.dino_model, "patch_tokens_from_output")
        with torch.inference_mode():
            for image in images:
                prepared = (
                    self.dino_model._imagenet_normalize(image) if has_patch_api else image
                )
                forward_start = time.perf_counter()
                if self.device.type == "cuda":
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                if has_patch_api:
                    tokens = self.dino_model.backbone.forward_features(prepared)
                else:
                    tokens = self.dino_model(prepared)
                    if tokens.ndim == 2:
                        tokens = tokens.unsqueeze(1)  # Mock global (B,C) -> local (B,1,C)
                if self.device.type == "cuda":
                    end.record()
                    forward_events.append((begin, end))
                else:
                    forward_times.append((time.perf_counter() - forward_start) * 1000)

                extract_start = time.perf_counter()
                if has_patch_api:
                    local = self.dino_model.patch_tokens_from_output(tokens).detach()
                else:
                    local = tokens.detach()
                patch_extract_ms += (time.perf_counter() - extract_start) * 1000
                pool_start = time.perf_counter()
                if self.device.type == "cuda":
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                global_feature = local.mean(dim=1, keepdim=True).detach()  # (B,1,C)
                if self.device.type == "cuda":
                    end.record()
                    pooling_events.append((begin, end))
                else:
                    pooling_times.append((time.perf_counter() - pool_start) * 1000)
                local_features.append(local)
                global_features.append(global_feature.squeeze(1))  # (B,C)

            stack_start = time.perf_counter()
            if self.device.type == "cuda":
                stack_begin = torch.cuda.Event(enable_timing=True)
                stack_end = torch.cuda.Event(enable_timing=True)
                stack_begin.record()
            global_timestep = torch.stack(global_features, dim=1).detach()  # (B,V,C)
            local_timestep = None
            if self.store_local_features:
                local_timestep = torch.stack(local_features, dim=1).detach()  # (B,V,N,C)
            if self.device.type == "cuda":
                stack_end.record()
                torch.cuda.synchronize(self.device)
                forward_times = [a.elapsed_time(b) for a, b in forward_events]
                pooling_times = [a.elapsed_time(b) for a, b in pooling_events]
                view_stack_ms = stack_begin.elapsed_time(stack_end)
            else:
                view_stack_ms = (time.perf_counter() - stack_start) * 1000

        timing = {
            "forward_times": forward_times,
            "patch_extract_ms": patch_extract_ms,
            "pooling_ms": sum(pooling_times),
            "view_stack_ms": view_stack_ms,
            "stage_wall_ms": (time.perf_counter() - stage_start) * 1000,
        }
        return global_timestep, local_timestep, timing


class MockDinoModel(nn.Module):
    def __init__(self, feature_dim=64, delay_ms=20.0):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)
        self.delay_seconds = delay_ms / 1000

    def forward(self, image):
        feature = image.float().mean(dim=(-2, -1))
        feature = self.projection(feature)
        time.sleep(self.delay_seconds)
        return feature


def main():
    dino_buffer = AsyncDinoBuffer(MockDinoModel(), device="cpu")
    dino_buffer.start()
    for frame_id in range(4000):
        capture_time = time.perf_counter()
        image_1 = torch.randn(1, 3, 64, 64)
        image_2 = torch.randn(1, 3, 64, 64)
        image_3 = torch.randn(1, 3, 64, 64)
        dino_buffer.submit_frame(
            frame_id, image_1, image_2, image_3, capture_time=capture_time
        )
        time.sleep(0.033)
    dino_buffer.stop()
    print("Buffer length:", len(dino_buffer.get_buffer()))
    print("Stats:", dino_buffer.get_stats())


if __name__ == "__main__":
    main()
