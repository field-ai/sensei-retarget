"""
Unitree G1 RobotConfig.

Joint names and limits sourced directly from:
  /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
"""
import pathlib
import numpy as np
from sensei.types import RobotConfig

_GMR_ROOT = pathlib.Path("/mnt/code/GMR")

# Joint names in MuJoCo model order (matches actuator order in XML)
G1_JOINT_NAMES: list[str] = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Lower limits (rad) — from g1_mocap_29dof.xml range attributes
G1_JOINT_LOWER = np.array([
    # Left leg
    -1.5700,  # left_hip_pitch
    -0.5236,  # left_hip_roll
    -1.5700,  # left_hip_yaw
    -0.0873,  # left_knee
    -0.8727,  # left_ankle_pitch
    -0.2618,  # left_ankle_roll
    # Right leg
    -1.5700,  # right_hip_pitch
    -1.5700,  # right_hip_roll
    -1.5700,  # right_hip_yaw
    -0.0873,  # right_knee
    -0.8727,  # right_ankle_pitch
    -0.2618,  # right_ankle_roll
    # Waist
    -1.5700,  # waist_yaw
    -0.5200,  # waist_roll
    -0.5200,  # waist_pitch
    # Left arm
    -3.0892,  # left_shoulder_pitch
    -0.6000,  # left_shoulder_roll
    -1.4000,  # left_shoulder_yaw
    -1.0472,  # left_elbow
    -1.9722,  # left_wrist_roll
    -1.6144,  # left_wrist_pitch
    -1.6144,  # left_wrist_yaw
    # Right arm
    -3.0892,  # right_shoulder_pitch
    -2.2515,  # right_shoulder_roll
    -2.0000,  # right_shoulder_yaw
    -1.0472,  # right_elbow
    -1.9722,  # right_wrist_roll
    -1.6144,  # right_wrist_pitch
    -1.6144,  # right_wrist_yaw
], dtype=np.float64)

# Upper limits (rad) — from g1_mocap_29dof.xml range attributes
G1_JOINT_UPPER = np.array([
    # Left leg
     1.5700,  # left_hip_pitch
     1.5700,  # left_hip_roll
     1.5700,  # left_hip_yaw
     2.8798,  # left_knee
     0.5236,  # left_ankle_pitch
     0.2618,  # left_ankle_roll
    # Right leg
     1.5700,  # right_hip_pitch
     0.5236,  # right_hip_roll
     1.5700,  # right_hip_yaw
     2.8798,  # right_knee
     0.5236,  # right_ankle_pitch
     0.2618,  # right_ankle_roll
    # Waist
     1.5700,  # waist_yaw
     0.5200,  # waist_roll
     0.5200,  # waist_pitch
    # Left arm
     1.1490,  # left_shoulder_pitch
     2.2515,  # left_shoulder_roll
     2.0000,  # left_shoulder_yaw
     1.7000,  # left_elbow
     1.9722,  # left_wrist_roll
     1.6144,  # left_wrist_pitch
     1.6144,  # left_wrist_yaw
    # Right arm
     1.1490,  # right_shoulder_pitch
     0.6000,  # right_shoulder_roll
     1.4000,  # right_shoulder_yaw
     1.7000,  # right_elbow
     1.9722,  # right_wrist_roll
     1.6144,  # right_wrist_pitch
     1.6144,  # right_wrist_yaw
], dtype=np.float64)

# Velocity limits (rad/s) — conservative; no explicit values in XML
G1_VEL_LIMITS = np.full(29, 10.0, dtype=np.float64)

G1_END_EFFECTORS: dict[str, str] = {
    "left_wrist":  "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_ankle":  "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
}


def get_g1_config() -> RobotConfig:
    """Return the canonical G1 RobotConfig for sensei."""
    mjcf = _GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    # URDF available in xr_teleoperate; populated in Phase 2 when Pinocchio is added
    urdf = (
        pathlib.Path(__file__).parent.parent.parent
        / "reference_repos" / "xr_teleoperate" / "assets" / "g1" / "g1_body29_hand14.urdf"
    )
    return RobotConfig(
        name="unitree_g1",
        mjcf_path=str(mjcf),
        urdf_path=str(urdf) if urdf.exists() else "",
        joint_names=G1_JOINT_NAMES,
        joint_lower=G1_JOINT_LOWER,
        joint_upper=G1_JOINT_UPPER,
        vel_limits=G1_VEL_LIMITS,
        end_effectors=G1_END_EFFECTORS,
        default_pose=np.zeros(29, dtype=np.float64),
        metadata={"gmr_robot_name": "unitree_g1"},
    )
