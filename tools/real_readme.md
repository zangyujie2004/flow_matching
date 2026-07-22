
  import time
  from collections import deque

  import numpy as np

  from infer.preprocess import build_state_frame, parse_preprocess_config
  from infer.runtime import FMInferenceRuntime

  runtime = FMInferenceRuntime(
      "/path/to/three_view_run_dir",
      device="cuda",
      warmup=False,
  )
  preprocess = parse_preprocess_config(runtime.cfg)
  buffer = runtime.start_async_dino()

  policy_frames = deque(maxlen=runtime.window_size)
  state_frames = deque(maxlen=68)  # 64 history + recent_frame=4

  for frame_id, frame in synchronized_camera_stream:
      # 最好来自驱动，并转换到与 perf_counter 相同的单调时钟域。
      capture_time = frame.capture_monotonic_s

    policy_frames.append(frame)
      state_frames.append(build_state_frame(frame, preprocess))

    # 在提交当前采样帧前使用截至 frame_id-4 的视觉历史。
      entries = buffer.get_buffer()
      expected_ids = list(range(frame_id - 64, frame_id, 4))
      actual_ids = [entry["frame_id"] for entry in entries]

    if (
          len(policy_frames) == runtime.window_size
          and len(state_frames) == 68
          and actual_ids == expected_ids
      ):
          # 对应训练时的 state offsets：frame_id-67 ... frame_id-4。
          memory_state_raw = np.stack(list(state_frames)[:-4])

    result = runtime.infer_from_window(
              list(policy_frames),
              preprocess=preprocess,
              memory_state_raw=memory_state_raw,
          )

    runtime.submit_async_dino_frame(
          frame_id,
          frame,
          capture_time=capture_time,
      )

  runtime.stop_async_dino()
