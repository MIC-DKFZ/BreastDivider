from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

MODEL_REPO_ID = "ykirchhoff/BreastDividerModel"
MODEL_REVISION = "main"
CHECKPOINT_NAME = "checkpoint_final.pth"
ALLOWED_MODEL_PATTERNS = [
    "*.json",
    "*.pkl",
    "*.pth",
    "fold_*/*",
    "dataset_fingerprint.json",
]


def download_model(
    download_dir: Optional[str | Path] = None,
    *,
    revision: str = MODEL_REVISION,
    force_download: bool = False,
) -> Path:
    """Download the published nnU-Net model from Hugging Face and return its path."""
    path = snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=revision,
        local_dir=(
            str(Path(download_dir).expanduser().resolve()) if download_dir else None
        ),
        allow_patterns=ALLOWED_MODEL_PATTERNS,
        force_download=force_download,
    )
    return Path(path)


def get_model_dir(
    model_dir: Optional[str | Path] = None,
    *,
    revision: str = MODEL_REVISION,
) -> Path:
    """Resolve an existing model directory or download the default model."""
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    return download_model(revision=revision)


class BreastDividerPredictor:
    """Small convenience wrapper around nnUNetPredictor."""

    def __init__(
        self,
        *,
        model_dir: Optional[str | Path] = None,
        revision: str = MODEL_REVISION,
        device: str = "cuda",
        folds: Optional[tuple[int, ...]] = None,
        checkpoint_name: str = CHECKPOINT_NAME,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
    ) -> None:
        self.model_dir = get_model_dir(model_dir, revision=revision)
        self.checkpoint_name = checkpoint_name
        self.folds = self._resolve_folds(folds)
        device = self._resolve_device(device)
        self._predictor = self._build_predictor(
            device=device,
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )

    def _resolve_folds(
        self, folds: Optional[tuple[int, ...]]
    ) -> Optional[tuple[int | str, ...]]:
        if folds is not None:
            return folds

        # Our published nnU-Net inference bundle only ships fold_all
        if (self.model_dir / "fold_all" / self.checkpoint_name).is_file():
            return ("all",)

        return None

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device != "cuda":
            return device

        if torch.cuda.is_available():
            return "cuda"

        print(
            "CUDA was requested but is not usable in the current environment. Falling back to CPU."
        )
        return "cpu"

    def _build_predictor(
        self,
        *,
        device: str,
        tile_step_size: float,
        use_gaussian: bool,
        use_mirroring: bool,
        perform_everything_on_device: bool,
        verbose: bool,
        verbose_preprocessing: bool,
        allow_tqdm: bool,
    ):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=torch.device(device),
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )
        predictor.initialize_from_trained_model_folder(
            str(self.model_dir),
            use_folds=self.folds,
            checkpoint_name=self.checkpoint_name,
        )
        return predictor

    def predict(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        overwrite: bool = True,
        save_probabilities: bool = False,
        num_processes_preprocessing: int = 2,
        num_processes_segmentation_export: int = 2,
        part_id: int = 0,
        num_parts: int = 1,
    ) -> Path:
        """Run inference for a single file or a nnU-Net style input directory."""
        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        if input_path.is_dir():
            if output_path.suffix:
                raise ValueError(
                    "Directory inference expects output_path to be a directory."
                )
            return self._predict_directory(
                input_path=input_path,
                output_path=output_path,
                overwrite=overwrite,
                save_probabilities=save_probabilities,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
                part_id=part_id,
                num_parts=num_parts,
            )

        return self._predict_single_file(
            input_path=input_path,
            output_path=output_path,
            overwrite=overwrite,
            save_probabilities=save_probabilities,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )

    def _predict_single_file(
        self,
        *,
        input_path: Path,
        output_path: Path,
        overwrite: bool,
        save_probabilities: bool,
        num_processes_preprocessing: int,
        num_processes_segmentation_export: int,
    ) -> Path:
        if not input_path.name.endswith(".nii.gz"):
            raise ValueError("Single-file inference expects a .nii.gz input file.")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        case_id = _strip_nii_gz(input_path.name)

        with (
            tempfile.TemporaryDirectory(prefix="breastdivider-input-") as tmp_input_dir,
            tempfile.TemporaryDirectory(
                prefix="breastdivider-output-"
            ) as tmp_output_dir,
        ):
            staged_input = Path(tmp_input_dir) / f"{case_id}_0000.nii.gz"
            staged_output = Path(tmp_output_dir) / f"{case_id}.nii.gz"
            shutil.copy2(input_path, staged_input)

            self._predictor.predict_from_files(
                str(Path(tmp_input_dir)),
                str(Path(tmp_output_dir)),
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
            )

            if not staged_output.exists():
                raise FileNotFoundError(
                    f"nnunetv2 did not produce the expected output file: {staged_output}"
                )

            shutil.move(str(staged_output), str(output_path))
            if save_probabilities:
                staged_probabilities = Path(tmp_output_dir) / f"{case_id}.npz"
                if staged_probabilities.exists():
                    probability_output = output_path.with_suffix("").with_suffix(".npz")
                    shutil.move(str(staged_probabilities), str(probability_output))
            return output_path

    def _predict_directory(
        self,
        *,
        input_path: Path,
        output_path: Path,
        overwrite: bool,
        save_probabilities: bool,
        num_processes_preprocessing: int,
        num_processes_segmentation_export: int,
        part_id: int,
        num_parts: int,
    ) -> Path:
        output_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="breastdivider-input-dir-"
        ) as tmp_input_dir:
            staged_input_dir = Path(tmp_input_dir)
            staged_any_files = False

            for input_file in sorted(input_path.iterdir()):
                if not input_file.is_file() or not input_file.name.endswith(".nii.gz"):
                    continue

                staged_any_files = True
                staged_name = self._normalize_case_filename(input_file.name)
                shutil.copy2(input_file, staged_input_dir / staged_name)

            if not staged_any_files:
                raise ValueError(
                    f"No .nii.gz files found in input directory: {input_path}"
                )

            self._predictor.predict_from_files(
                str(staged_input_dir),
                str(output_path),
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
                part_id=part_id,
                num_parts=num_parts,
            )

        return output_path

    @staticmethod
    def _normalize_case_filename(filename: str) -> str:
        case_id = _strip_nii_gz(filename)
        if re.search(r"_\d{4}$", case_id):
            return f"{case_id}.nii.gz"
        return f"{case_id}_0000.nii.gz"


def predict(
    *,
    input_path: str | Path,
    output_path: str | Path,
    model_dir: Optional[str | Path] = None,
    revision: str = MODEL_REVISION,
    device: str = "cuda",
    folds: Optional[tuple[int, ...]] = None,
    checkpoint_name: str = CHECKPOINT_NAME,
    overwrite: bool = True,
    save_probabilities: bool = False,
    num_processes_preprocessing: int = 2,
    num_processes_segmentation_export: int = 2,
) -> Path:
    """One-shot prediction helper."""
    predictor = BreastDividerPredictor(
        model_dir=model_dir,
        revision=revision,
        device=device,
        folds=folds,
        checkpoint_name=checkpoint_name,
    )
    return predictor.predict(
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        save_probabilities=save_probabilities,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
    )


def _strip_nii_gz(filename: str) -> str:
    if not filename.endswith(".nii.gz"):
        return Path(filename).stem
    return filename[: -len(".nii.gz")]
