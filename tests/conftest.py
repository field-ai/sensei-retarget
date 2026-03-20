"""
Shared pytest fixtures.
"""
import pathlib
import numpy as np
import pytest

from sensei.types import MotionSequence, RobotConfig

# Pre-processed GVHMR outputs (already on disk, no inference needed)
GVHMR_CLIPS = {
    "tennis":       "/mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    "basketball":   "/mnt/code/GVHMR/outputs/demo/basketball_clip/hmr4d_results.pt",
    "dance":        "/mnt/code/GVHMR/outputs/demo/dance_clip/hmr4d_results.pt",
    "0_input":      "/mnt/code/GVHMR/outputs/demo/0_input_video/hmr4d_results.pt",
}


@pytest.fixture
def tennis_pt_path() -> str:
    return GVHMR_CLIPS["tennis"]


@pytest.fixture
def all_clip_paths() -> dict[str, str]:
    return GVHMR_CLIPS


@pytest.fixture
def mock_motion_sequence() -> MotionSequence:
    """Minimal MotionSequence for unit tests that don't need real data."""
    N, J = 10, 21
    return MotionSequence(
        fps=30.0,
        num_frames=N,
        num_joints=J,
        body_pose=np.zeros((N, J, 3), dtype=np.float64),
        global_orient=np.zeros((N, 3), dtype=np.float64),
        transl=np.zeros((N, 3), dtype=np.float64),
        betas=np.zeros(10, dtype=np.float64),
        joint_names=[f"joint_{i}" for i in range(J)],
        landmarks=None,
    )


@pytest.fixture
def g1_config() -> RobotConfig:
    from sensei.robots.g1 import get_g1_config
    return get_g1_config()
