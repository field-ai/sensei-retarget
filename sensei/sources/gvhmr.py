"""
GVHMRSource: loads GVHMR hmr4d_results.pt files → MotionSequence.

Requires (in the sensei conda env):
    pip install -e third_party/GMR   # provides general_motion_retargeting
    pip install torch                # for torch.load on .pt files

The SMPL-X body model must exist at:
    /mnt/code/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
"""
from __future__ import annotations

import pathlib
import numpy as np

from sensei.base.source import MotionSource
from sensei.types import MotionSequence

_DEFAULT_SMPLX_FOLDER = pathlib.Path("/mnt/code/GMR/assets/body_models")
_DEFAULT_TARGET_FPS = 30


class GVHMRSource(MotionSource):
    """
    Loads a GVHMR hmr4d_results.pt file and converts it to MotionSequence.

    Delegates SMPL-X FK to GMR's utilities (load_gvhmr_pred_file,
    get_gvhmr_data_offline_fast) so we don't duplicate that logic.

    The resulting MotionSequence.landmarks list is in GMR's native format:
        landmarks[i] = {joint_name: (pos (3,), rot (Rotation))}
    and is consumed directly by GMRSolver.

    Args:
        smplx_folder: path containing SMPLX body model files.
        target_fps:   downsample to this FPS via SLERP (default 30).
    """

    def __init__(
        self,
        smplx_folder: str | pathlib.Path = _DEFAULT_SMPLX_FOLDER,
        target_fps: int = _DEFAULT_TARGET_FPS,
    ) -> None:
        self._smplx_folder = pathlib.Path(smplx_folder)
        self._target_fps = target_fps

    @property
    def name(self) -> str:
        return "gvhmr"

    def can_load(self, path: str) -> bool:
        p = pathlib.Path(path)
        return p.suffix == ".pt" and p.exists()

    def load(self, path: str) -> MotionSequence:
        """
        Load a GVHMR .pt file and return a MotionSequence.

        Runs SMPL-X FK to populate .landmarks (required by GMRSolver).
        Converts all tensors to float64 numpy.
        """
        # Late imports — only fail if GMR / smplx not installed
        from general_motion_retargeting.utils.smpl import (
            load_gvhmr_pred_file,
            get_gvhmr_data_offline_fast,
            JOINT_NAMES,
        )

        smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
            path, self._smplx_folder
        )

        frames_list, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=self._target_fps
        )

        N = len(frames_list)

        # Convert raw SMPL params to float64; slice to aligned frame count
        body_pose = smplx_data["pose_body"].astype(np.float64)       # (N_orig, 63)
        global_orient = smplx_data["root_orient"].astype(np.float64)  # (N_orig, 3)
        transl = smplx_data["trans"].astype(np.float64)               # (N_orig, 3)
        betas = smplx_data["betas"].astype(np.float64).flatten()[:10] # (10,)

        body_pose = body_pose.reshape(-1, 21, 3)[:N]   # (N, 21, 3)
        global_orient = global_orient[:N]               # (N, 3)
        transl = transl[:N]                             # (N, 3)

        joint_names = list(JOINT_NAMES[:21])

        return MotionSequence(
            fps=float(aligned_fps),
            num_frames=N,
            num_joints=21,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            betas=betas,
            joint_names=joint_names,
            landmarks=frames_list,
            metadata={
                "source_path": str(path),
                "human_height_m": float(human_height),
                "smplx_folder": str(self._smplx_folder),
            },
        )
