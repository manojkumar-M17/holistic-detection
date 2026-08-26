import time
import config.config as cfg

class ResourceMonitor:
    """
    ResourceMonitor tracks system performance (CPU usage, memory footprint, GPU usage, FPS, 
    and processing latency) to prevent hardware thermal throttling or system crashes.
    Supports dynamic profiling adjustments (e.g. lowering resolution, skipping frames, 
    or disabling specific heavy pipelines like hand landmarks) under heavy load.
    """
    def __init__(self):
        pass

    def get_system_metrics(self) -> dict:
        """
        Retrieves active hardware utilization rates (CPU, RAM, GPU, thermals).
        """
        pass

    def calculate_fps(self) -> float:
        """
        Computes the current frame processing rate (FPS) of the main proctoring loop.
        """
        pass

    def check_thresholds_and_adapt(self) -> dict:
        """
        Checks system load against predefined safety thresholds and returns recommended 
        pipeline adaptations (e.g., skip next frame, lower MediaPipe complexity, etc.).
        """
        pass
