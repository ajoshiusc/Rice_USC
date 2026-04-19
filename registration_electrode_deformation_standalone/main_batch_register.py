import os

from electrode_label_deformation import deform_subject_labels, load_electrode_trajectories
from register_exvivo import run_registration

subjects = ["r52", "r53","r54", "r57", "r59", "r60"]


RodentToolsDir = "/home/ajoshi/project2_ajoshi_1183/data/RodentTools"

if not os.path.exists(RodentToolsDir):
    RodentToolsDir = "/project2/ajoshi_1183/data/RodentTools"

RAW_BASE = f"{RodentToolsDir}/for_Seymour/11_15_2025/MRI/Raw T2/"
OUT_BASE = f"{RodentToolsDir}/for_Seymour/11_15_2025/MRI/CompletedAtlases_new_04_17_2026/"

atlas_bse_t2 = f"{RodentToolsDir}/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_T2star_v1.01.bse.nii.gz"
atlas_labels = f"{RodentToolsDir}/Atlases/Waxholm/WHS_SD_rat_atlas_v4.01.nii.gz"
TRAJECTORY_WORKBOOK = "/home/ajoshi/Downloads/Acute Rat Probe Trajectories.xlsx"


def apply_electrode_deformation(subj, output_dir, trajectories_by_subject):
    labels_path = os.path.join(output_dir, f"atlas_to_{subj}.reoriented.bse_reg.nonlin.label.nii.gz")
    output_path = os.path.join(output_dir, f"atlas_to_{subj}.reoriented.bse_reg.nonlin.electrode_deformed.label.nii.gz")
    mask_path = os.path.join(output_dir, f"atlas_to_{subj}.reoriented.bse_reg.nonlin.electrode_mask.nii.gz")

    if subj not in trajectories_by_subject:
        print(f"Skipping electrode deformation for {subj}: no trajectories found in {TRAJECTORY_WORKBOOK}")
        return

    if not os.path.exists(labels_path):
        print(f"Skipping electrode deformation for {subj}: labels not found at {labels_path}")
        return

    deform_subject_labels(
        subject_id=subj,
        labels_path=labels_path,
        trajectories_by_subject=trajectories_by_subject,
        output_path=output_path,
        electrode_mask_output_path=mask_path,
    )
    print(f"Electrode-deformed labels saved to {output_path}")

if __name__ == "__main__":
    trajectories_by_subject = {}
    if os.path.exists(TRAJECTORY_WORKBOOK):
        trajectories_by_subject = load_electrode_trajectories(TRAJECTORY_WORKBOOK)
    else:
        print(f"Trajectory workbook not found: {TRAJECTORY_WORKBOOK}")

    for subj in subjects:
        sub_bse_t2 = f"{RAW_BASE}{subj}/{subj}.reoriented.bse.nii.gz"
        outdir = f"{OUT_BASE}{subj}"
        final_labels_path = os.path.join(outdir, f"atlas_to_{subj}.reoriented.bse_reg.nonlin.label.nii.gz")
        print(f"\n--- Processing {subj} ---")
        if os.path.exists(final_labels_path):
            print(f"Skipping registration for {subj}: found existing final labels at {final_labels_path}")
        else:
            run_registration(sub_bse_t2, atlas_bse_t2, atlas_labels, outdir, skip_affine=False)
        apply_electrode_deformation(subj, outdir, trajectories_by_subject)
