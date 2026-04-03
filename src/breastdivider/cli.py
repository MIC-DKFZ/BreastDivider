from __future__ import annotations

import argparse
from pathlib import Path

import torch

from breastdivider.api import BreastDividerPredictor, download_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breastdivider",
        description="Run BreastDivider left/right breast MRI segmentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download the published BreastDivider model from Hugging Face.",
    )
    download_parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Optional directory where the model snapshot should be stored.",
    )
    download_parser.add_argument(
        "--revision",
        default="main",
        help="Model repository revision, branch, or tag to download from Hugging Face.",
    )
    download_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the model files even if they are already available locally.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on a single .nii.gz file or a nnU-Net style input directory.",
    )
    predict_parser.add_argument(
        "input_path",
        type=Path,
        help="Input .nii.gz file or nnU-Net style input directory containing the cases to segment.",
    )
    predict_parser.add_argument(
        "output_path",
        type=Path,
        help="Output segmentation file for single-file mode or output directory for batch mode.",
    )
    predict_parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to an existing downloaded model directory. Defaults to auto-downloading the model.",
    )
    predict_parser.add_argument(
        "--revision",
        default="main",
        help="Model repository revision, branch, or tag to use when auto-downloading weights.",
    )
    predict_parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        type=str,
        choices=["cpu", "cuda"],
        help="Inference device. Defaults to 'cuda' when available, otherwise 'cpu'.",
    )
    predict_parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.pth",
        help="Checkpoint filename inside the nnU-Net model directory to load for inference.",
    )
    predict_parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also export class probability files alongside the segmentation output when supported.",
    )
    predict_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of overwriting existing output files.",
    )
    predict_parser.add_argument(
        "--num-processes-preprocessing",
        type=int,
        default=2,
        help="Number of worker processes used by nnU-Net for preprocessing inputs.",
    )
    predict_parser.add_argument(
        "--num-processes-export",
        type=int,
        default=2,
        help="Number of worker processes used by nnU-Net for exporting segmentation outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        model_dir = download_model(
            download_dir=args.download_dir,
            revision=args.revision,
            force_download=args.force_download,
        )
        print(model_dir)
        return

    predictor = BreastDividerPredictor(
        model_dir=args.model_dir,
        revision=args.revision,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
    )
    output = predictor.predict(
        input_path=args.input_path,
        output_path=args.output_path,
        overwrite=not args.no_overwrite,
        save_probabilities=args.save_probabilities,
        num_processes_preprocessing=args.num_processes_preprocessing,
        num_processes_segmentation_export=args.num_processes_export,
    )
    print(output)
