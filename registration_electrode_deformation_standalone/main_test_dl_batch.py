subjects = [
    {
        "name": "r52",
        "sub_bse_t2": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Raw T2/r52/r52.reoriented.bse.nii.gz",
        "outdir": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Completed Atlases/r52",
    },
    {
        "name": "r53",
        "sub_bse_t2": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Raw T2/r53/r53.reoriented.bse.nii.gz",
        "outdir": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Completed Atlases/r53",
    },
    {
        "name": "r57",
        "sub_bse_t2": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Raw T2/r57/r57.reoriented.bse.nii.gz",
        "outdir": "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Completed Atlases/r57",
    },
]

atlas_bse_t2 = "/deneb_disk/RodentTools/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_T2star_v1.01.bse.nii.gz"
atlas_labels = "/deneb_disk/RodentTools/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_atlas_v4.nii.gz"


def run_test(sub_bse_t2, output_dir):
    from register_exvivo import run_registration
    run_registration(sub_bse_t2, atlas_bse_t2, atlas_labels, output_dir, skip_affine=False)


if __name__ == "__main__":
    for subj in subjects:
        print(f"\n--- Processing {subj['name']} ---")
        run_test(subj["sub_bse_t2"], subj["outdir"])
