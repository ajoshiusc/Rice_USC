# deform_image_by_electrode.py

import numpy as np
import SimpleITK as sitk
from foreign_object_utils import (
    compute_affine_from_2pts,
    apply_affine_to_image,
    apply_affine_to_deformation_field,
    apply_deformation_field_to_image_itk
)

def deform_image_by_electrode(
    target_path,
    target_electrode_path,
    target_electrode_deformed_path,
    template_cyl_path='cylinder_mask_0.1mm.nii.gz',
    template_deformation_field_path='deformation_field_0.1mm_itk.nii.gz',
    template_pts=[[0, 0, 0], [0, 0, -8]],
    target_pts=[[5.8, 1.2, 4], [4.17, 1.87, 2.45]],
    output_path="deformed_template.nii.gz",
    output_deformation_field_path="deformed_deformation_field_itk.nii.gz",
    islabel=False,
):
    """
    Deform a target image by electrode using a template and deformation field.
    """
    if islabel:
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear
    # Step 1: Compute affine from points and apply to template image
    affine = compute_affine_from_2pts(template_pts, target_pts)
    apply_affine_to_image(template_cyl_path, affine, output_path)

    # Step 2: Apply affine to deformation field
    apply_affine_to_deformation_field(
        deformation_field_path=template_deformation_field_path,
        affine=affine,
        output_path=output_deformation_field_path
    )

    # Step 3: Create mask from deformed template and apply to target
    img = sitk.ReadImage(output_path, sitk.sitkFloat32)
    msk = (img - 1.0)
    msk[msk < 0.0] = 0
    #msk = (1.0 - msk) > 0.5
    #msk = sitk.Cast(msk, sitk.sitkFloat32)
    #msk = sitk.BinaryThresholdImageFilter().Execute(msk, 0.5, 1.0)
    #msk = sitk.Cast(msk, sitk.sitkFloat32)
    
    target = sitk.ReadImage(target_path, sitk.sitkFloat32)

    # Resample mask to match target
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    msk_resampled = resampler.Execute(msk)
    msk_resampled = (1.0 - msk_resampled)
    msk_resampled[msk_resampled < 0.0] = 0

    if islabel:
        msk_resampled = msk_resampled > 0.5

    msk_resampled = sitk.Cast(msk_resampled, sitk.sitkFloat32)

    target_masked = target * msk_resampled
    sitk.WriteImage(target_masked, target_electrode_path)

    # Step 4: Apply deformation field to masked target
    apply_deformation_field_to_image_itk(
        target_electrode_path,
        output_deformation_field_path,
        target_electrode_deformed_path,
        interp=interp
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python deform_image_by_electrode.py <target_path> <target_electrode_path> <target_electrode_deformed_path>")
        sys.exit(1)
    deform_image_by_electrode(
        target_path=sys.argv[1],
        target_electrode_path=sys.argv[2],
        target_electrode_deformed_path=sys.argv[3]
    )