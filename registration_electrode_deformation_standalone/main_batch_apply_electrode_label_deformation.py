from pathlib import Path

from electrode_label_deformation import deform_subject_labels, load_electrode_trajectories


subjects = ["r52", "r53", "r54", "r57", "r59", "r60"]


RodentToolsDir = "/home/ajoshi/project2_ajoshi_1183/data/RodentTools"

if not Path(RodentToolsDir).exists():
    RodentToolsDir = "/project2/ajoshi_1183/data/RodentTools"

OUT_BASE = Path(f"{RodentToolsDir}/for_Seymour/11_15_2025/MRI/CompletedAtlases_new_04_17_2026/")
WORKBOOK = Path("/home/ajoshi/Downloads/Acute Rat Probe Trajectories.xlsx")


def deform_subject(subject: str, workbook_trajectories: dict[str, list]) -> None:
    labels_path = OUT_BASE / subject / f"atlas_to_{subject}.reoriented.bse_reg.nonlin.label.nii.gz"
    output_path = OUT_BASE / subject / f"atlas_to_{subject}.reoriented.bse_reg.nonlin.electrode_deformed.label.nii.gz"
    mask_output_path = OUT_BASE / subject / f"atlas_to_{subject}.reoriented.bse_reg.nonlin.electrode_mask.nii.gz"

    if subject not in workbook_trajectories:
        print(f"Skipping {subject}: no electrode trajectories found in {WORKBOOK}")
        return

    if not labels_path.exists():
        print(f"Skipping {subject}: labels not found at {labels_path}")
        return

    deform_subject_labels(
        subject_id=subject,
        labels_path=labels_path,
        trajectories_by_subject=workbook_trajectories,
        output_path=output_path,
        electrode_mask_output_path=mask_output_path,
    )
    print(f"Finished {subject}: {output_path}")


if __name__ == "__main__":
    trajectories = load_electrode_trajectories(WORKBOOK)
    for subject in subjects:
        print(f"\n--- Electrode deformation for {subject} ---")
        deform_subject(subject, trajectories)
