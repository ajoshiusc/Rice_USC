import os


subjects = ["r52", "r53","r54", "r57", "r59", "r60"]


RodentToolsDir = "/home/ajoshi/project2_ajoshi_1183/data/RodentTools"

if not os.path.exists(RodentToolsDir):
    RodentToolsDir = "/project2/ajoshi_1183/data/RodentTools"

RAW_BASE = f"{RodentToolsDir}/for_Seymour/11_15_2025/MRI/Raw T2/"
OUT_BASE = f"{RodentToolsDir}/for_Seymour/11_15_2025/MRI/CompletedAtlases_new_11_29_2025/"

atlas_bse_t2 = f"{RodentToolsDir}/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_T2star_v1.01.bse.nii.gz"
atlas_labels = f"{RodentToolsDir}/Atlases/Waxholm/WHS_SD_rat_atlas_v4.01.nii.gz"


def run_test(sub_bse_t2, output_dir):
    from register_exvivo import run_registration
    run_registration(sub_bse_t2, atlas_bse_t2, atlas_labels, output_dir, skip_affine=False)

if __name__ == "__main__":
    for subj in subjects:
        sub_bse_t2 = f"{RAW_BASE}{subj}/{subj}.reoriented.bfc.nii.gz"
        outdir = f"{OUT_BASE}{subj}"
        print(f"\n--- Processing {subj} ---")
        run_test(sub_bse_t2, outdir)
