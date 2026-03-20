"""
Integration tests for GVHMRSource.
Requires: pip install -e third_party/GMR  (mink, mujoco, smplx, torch)
"""
import numpy as np
import pytest
from sensei.sources.gvhmr import GVHMRSource

pytestmark = pytest.mark.integration


# ── can_load ──────────────────────────────────────────────────────────────────

def test_can_load_rejects_pkl():
    assert not GVHMRSource().can_load("motion.pkl")

def test_can_load_rejects_nonexistent_pt():
    assert not GVHMRSource().can_load("/tmp/nonexistent.pt")

def test_can_load_accepts_real_pt(tennis_pt_path):
    assert GVHMRSource().can_load(tennis_pt_path)


# ── load ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tennis_motion(tennis_pt_path):
    return GVHMRSource().load(tennis_pt_path)


def test_load_fps(tennis_motion):
    assert tennis_motion.fps == pytest.approx(30.0)

def test_load_frame_count(tennis_motion):
    assert tennis_motion.num_frames > 0

def test_load_dtypes(tennis_motion):
    assert tennis_motion.body_pose.dtype == np.float64
    assert tennis_motion.global_orient.dtype == np.float64
    assert tennis_motion.transl.dtype == np.float64
    assert tennis_motion.betas.dtype == np.float64

def test_load_shapes(tennis_motion):
    N, J = tennis_motion.num_frames, tennis_motion.num_joints
    assert tennis_motion.body_pose.shape    == (N, J, 3)
    assert tennis_motion.global_orient.shape == (N, 3)
    assert tennis_motion.transl.shape        == (N, 3)
    assert tennis_motion.betas.shape         == (10,)

def test_load_landmarks_present(tennis_motion):
    assert tennis_motion.landmarks is not None
    assert len(tennis_motion.landmarks) == tennis_motion.num_frames

def test_load_landmarks_have_root(tennis_motion):
    # GMR uses SMPL-X joint names; root is "pelvis"
    frame = tennis_motion.landmarks[0]
    root_keys = [k for k in frame if "pelvis" in k.lower() or "hips" in k.lower() or "root" in k.lower()]
    assert len(root_keys) >= 1, f"No root joint found. Keys: {list(frame.keys())[:10]}"
    pos, _ = frame[root_keys[0]]
    assert pos.shape == (3,)

def test_load_landmarks_have_wrists(tennis_motion):
    frame = tennis_motion.landmarks[0]
    wrist_keys = [k for k in frame if "wrist" in k.lower()]
    assert len(wrist_keys) >= 2, f"Expected wrist landmarks, got: {list(frame.keys())}"

def test_load_metadata(tennis_motion):
    assert "human_height_m" in tennis_motion.metadata
    h = tennis_motion.metadata["human_height_m"]
    assert 1.4 < h < 2.2, f"Implausible human height: {h}"

def test_betas_shape_10(tennis_motion):
    assert tennis_motion.betas.shape == (10,)


# ── All four clips load without error ─────────────────────────────────────────

@pytest.mark.parametrize("clip", ["tennis", "basketball", "dance", "0_input"])
def test_all_clips_load(all_clip_paths, clip):
    path = all_clip_paths[clip]
    motion = GVHMRSource().load(path)
    assert motion.num_frames > 0
    assert motion.landmarks is not None
