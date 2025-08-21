import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from foreign_object_utils import (
    create_cylindrical_deformation_field,
    save_deformation_field_nifti,
    save_deformation_field_itk,
    create_cylinder_mask,
    save_nifti_image,
    apply_deformation_field_to_image_itk,
    apply_deformation_field_to_image_scipy
)




if __name__ == "__main__":
    # Parameters
    voxel_size = 0.1  # mm
    outer_cylinder_diameter = 1.60  # mm
    inner_cylinder_diameter = 2*0.20  # mm
    z_cutoff_mm = 8.0  # mm, optional, can be None for full height
    expansion_factor = 1.0  # No expansion, can be adjusted if needed


    z_cutoff = int(z_cutoff_mm / voxel_size) if z_cutoff_mm is not None else None

    outer_cylinder_radius = outer_cylinder_diameter / 2.0
    inner_cylinder_radius = inner_cylinder_diameter / 2.0
    volume_size = 32.0  # mm
    num_voxels = int(volume_size / voxel_size)
    center = num_voxels // 2
    outer_radius_voxels = int(outer_cylinder_radius / voxel_size)
    inner_radius_voxels = int(inner_cylinder_radius / voxel_size)

    # Affine for NIfTI
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = voxel_size
    affine[0, 3] = affine[1, 3] = affine[2, 3] = -center * voxel_size

    # Create and save deformation field
    deformation_field = create_cylindrical_deformation_field(
        num_voxels, center, inner_radius_voxels, outer_radius_voxels, inner_cylinder_radius, voxel_size, z_cutoff=z_cutoff, expansion=expansion_factor)
    save_deformation_field_nifti(deformation_field, affine, "deformation_field_0.1mm.nii.gz")
    save_deformation_field_itk(deformation_field, voxel_size, center, "deformation_field_0.1mm_itk.nii.gz")

    # Create and save cylinder mask
    cylinder_mask = create_cylinder_mask(num_voxels, center, inner_radius_voxels, outer_radius_voxels, z_cutoff=z_cutoff)
    save_nifti_image(cylinder_mask, affine, "cylinder_mask_0.1mm.nii.gz")

    # Apply deformation field to cylinder mask (using ITK field)
    apply_deformation_field_to_image_itk(
        "cylinder_mask_0.1mm.nii.gz",
        "deformation_field_0.1mm_itk.nii.gz",
        "deformed_cylinder_mask_0.1mm_itk.nii.gz",
        interp=sitk.sitkLinear  # Use linear interpolation for smoother results
    )


    apply_deformation_field_to_image_scipy(
        "cylinder_mask_0.1mm.nii.gz",
        "deformation_field_0.1mm.nii.gz",
        "deformed_cylinder_mask_0.1mm_scipy.nii.gz"
    )


