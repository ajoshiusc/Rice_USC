import argparse
from pathlib import Path

from electrode_label_deformation import deform_subject_labels_from_workbook


DEFAULT_WORKBOOK = "/home/ajoshi/Downloads/Acute Rat Probe Trajectories.xlsx"


def _default_output_path(labels_path: Path) -> Path:
    name = labels_path.name
    if name.endswith(".nii.gz"):
        return labels_path.with_name(name[:-7] + ".electrode_deformed.nii.gz")
    if name.endswith(".nii"):
        return labels_path.with_name(name[:-4] + ".electrode_deformed.nii")
    return labels_path.with_name(name + ".electrode_deformed")


def _default_mask_output_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".nii.gz"):
        return output_path.with_name(name[:-7] + ".electrode_mask.nii.gz")
    if name.endswith(".nii"):
        return output_path.with_name(name[:-4] + ".electrode_mask.nii")
    return output_path.with_name(name + ".electrode_mask")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply subject-specific electrode deformations to a label map.")
    parser.add_argument("--subject", required=True, help="Subject identifier such as r52 or r59.")
    parser.add_argument("--labels", required=True, help="Path to the registered subject-space label map.")
    parser.add_argument(
        "--workbook",
        default=DEFAULT_WORKBOOK,
        help=f"Path to the XLSX workbook with electrode trajectories. Default: {DEFAULT_WORKBOOK}",
    )
    parser.add_argument("--output", help="Output path for the deformed label map.")
    parser.add_argument(
        "--electrode-mask-output",
        help="Optional output path for a combined binary mask of all modeled electrodes.",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_path = Path(args.output) if args.output else _default_output_path(labels_path)
    electrode_mask_output = (
        Path(args.electrode_mask_output)
        if args.electrode_mask_output
        else _default_mask_output_path(output_path)
    )

    deformed_path = deform_subject_labels_from_workbook(
        subject_id=args.subject,
        labels_path=labels_path,
        workbook_path=args.workbook,
        output_path=output_path,
        electrode_mask_output_path=electrode_mask_output,
    )
    print(f"Deformed labels written to: {deformed_path}")
    print(f"Electrode mask written to: {electrode_mask_output}")


if __name__ == "__main__":
    main()
