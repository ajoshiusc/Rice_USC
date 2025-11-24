sub_bse_t2 = "/deneb_disk/RodentTools/for_Seymour/07_30_2025_R57/R57.reoriented.nii.gz" #subbase+".bfc.nii.gz"
outdir = "/deneb_disk/RodentTools/for_Seymour/R57_nobse/registration_test_output"
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

