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
    ):
        self.dino_model = dino_model.to(device)
        self.dino_model.eval()
        self.device = torch.device(device)
        self.sample_interval_frames = sample_interval_frames
        self.deadline_ms = deadline_ms
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

    def get_stats(self):
        with self.lock:
            timing_lists = {
                "forward_1_ms": list(self.forward_1_times),
                "forward_2_ms": list(self.forward_2_times),
                "forward_3_ms": list(self.forward_3_times),
                "gpu_total_ms": list(self.gpu_total_times),
                "stage_wall_ms": list(self.stage_wall_times),
                "end_to_end_ms": list(self.end_to_end_times),
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
            md, forward_times, gpu_total_ms, stage_wall_ms = self._run_dino(images)
            ready_time = time.perf_counter()
            entry = {"frame_id": frame_id, "feature": md,
                "capture_time": capture_time, "ready_time": ready_time,
                "stage_wall_ms": stage_wall_ms, "end_to_end_ms": (ready_time - capture_time) * 1000,
            }
            with self.lock:
                self.buffer.append(entry)
                self.forward_1_times.append(forward_times[0])
                self.forward_2_times.append(forward_times[1])
                if len(forward_times) == 3:
                    self.forward_3_times.append(forward_times[2])
                self.gpu_total_times.append(gpu_total_ms)
                self.stage_wall_times.append(stage_wall_ms)
                self.end_to_end_times.append(entry["end_to_end_ms"])
                self.processed_count += 1
                self.deadline_miss_count += stage_wall_ms > self.deadline_ms

    def _run_dino(self, images):
        stage_start = time.perf_counter()
        images = [image.to(self.device) for image in images]
        features = []
        forward_times = []
        cuda_events = []
        with torch.inference_mode():
            for image in images:
                start = time.perf_counter()
                if self.device.type == "cuda":
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                features.append(self.dino_model(image))
                if self.device.type == "cuda":
                    end.record()
                    cuda_events.append((begin, end))
                else:
                    forward_times.append((time.perf_counter() - start) * 1000)
            md = torch.stack(features).mean(dim=0).detach()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                forward_times = [begin.elapsed_time(end) for begin, end in cuda_events]
        stage_wall_ms = (time.perf_counter() - stage_start) * 1000
        return md, forward_times, sum(forward_times), stage_wall_ms


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
