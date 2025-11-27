sub_bse_t2 = "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Raw T2/r52/r52.reoriented.bse.nii.gz" #subbase+".bfc.nii.gz"
outdir = "/home/ajoshi/project2_ajoshi_1183/data/RodentTools/for_Seymour/11_15_2025/MRI/Completed Atlases/r52_128_gemini_13"


atlas_bse_t2 = "/deneb_disk/RodentTools/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_T2star_v1.01.bse.nii.gz"
atlas_labels = "/deneb_disk/RodentTools/Atlases/Waxholm/WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_atlas_v4.nii.gz"


def run_test(output_dir=outdir):
    """Run the registration using the test paths defined above.

    This imports `register_exvivo.run_registration` lazily to avoid import
    cycles when `register_exvivo.py` imports this module.
    """
    from register_exvivo import run_registration

    run_registration(sub_bse_t2, atlas_bse_t2, atlas_labels, output_dir, skip_affine=False)


if __name__ == "__main__":
    run_test()

