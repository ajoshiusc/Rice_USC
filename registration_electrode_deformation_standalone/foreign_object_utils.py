import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import map_coordinates



def create_cylindrical_deformation_field(
    num_voxels, center, inner_radius_voxels, outer_radius_voxels, inner_cylinder_radius, voxel_size, z_cutoff=None, expansion = 1.0):
    """Create a 3D cylindrical deformation field (X, Y, Z, 3)"""
    # 2D radial field
    deformation_field_2d = np.zeros((num_voxels, num_voxels, 2), dtype=np.float32)
    x, y = np.meshgrid(np.arange(num_voxels), np.arange(num_voxels), indexing='ij')
    dx = x - center
    dy = y - center
    distance_from_center = np.sqrt(dx**2 + dy**2)
    radial_direction = -np.stack((dx, dy), axis=-1) / (distance_from_center[..., None] + 1e-8)

    # Inner cylinder
    inner_mask = distance_from_center <= inner_radius_voxels
    deformation_field_2d[inner_mask] = expansion * radial_direction[inner_mask] * (inner_cylinder_radius / voxel_size)

    # Outer cylinder
    outer_mask = (distance_from_center > inner_radius_voxels) & (distance_from_center <= outer_radius_voxels)
    distance_from_inner_surface = (distance_from_center - inner_radius_voxels)[outer_mask]
    max_deformation_voxels = expansion*(inner_cylinder_radius / voxel_size)
    linear_deformation = max_deformation_voxels * (1 - (distance_from_inner_surface / (outer_radius_voxels - inner_radius_voxels)))
    deformation_field_2d[outer_mask] = radial_direction[outer_mask] * linear_deformation[..., None]

    # Stack along z, add zero z-component
    deformation_field = np.repeat(deformation_field_2d[:, :, np.newaxis, :], num_voxels, axis=2)
    deformation_field = np.concatenate(
        (deformation_field, np.zeros((num_voxels, num_voxels, num_voxels, 1), dtype=np.float32)), axis=3
    )

    # Apply z cutoff if specified
    if z_cutoff is not None:
        z = np.arange(num_voxels) - center
        z_mask = np.abs(z[np.newaxis, np.newaxis,:]) <= z_cutoff
        deformation_field *= z_mask[..., np.newaxis]

    return deformation_field

def save_deformation_field_nifti(deformation_field, affine, filename):
    """Save deformation field as NIfTI (X, Y, Z, 3)"""
    nifti_img = nib.Nifti1Image(deformation_field, affine)
    nib.save(nifti_img, filename)
    print(f"NIfTI file saved as '{filename}'")

def save_deformation_field_itk(deformation_field, voxel_size, center, filename):
    """Save deformation field as ITK/SimpleITK vector field (ZYX order, mm, ITK conventions)"""
    # Permute to (Z, Y, X, 3)
    def_fld_itk = np.transpose(deformation_field, (2, 1, 0, 3)).copy()
    def_fld_itk *= voxel_size  # Convert to mm
    def_fld_itk[..., 0] *= -1  # Invert Z
    def_fld_itk[..., 1] *= -1  # Invert Y

    sitk_def = sitk.GetImageFromArray(def_fld_itk, isVector=True)
    sitk_def.SetSpacing((voxel_size, voxel_size, voxel_size))
    sitk_def.SetOrigin((center * voxel_size, center * voxel_size, -center * voxel_size))
    sitk_def.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    sitk.WriteImage(sitk_def, filename)
    print(f"ITK file saved as '{filename}'")

def create_cylinder_mask(num_voxels, center, inner_radius_voxels, outer_radius_voxels, z_cutoff=None):
    """Create a 3D mask with two concentric cylinders"""
    x, y, z = np.meshgrid(
        np.arange(num_voxels), np.arange(num_voxels), np.arange(num_voxels), indexing='ij'
    )
    distance_squared = (x - center) ** 2 + (y - center) ** 2
    mask = np.zeros((num_voxels, num_voxels, num_voxels), dtype=np.float32)
    mask[distance_squared <= inner_radius_voxels**2] = 1.0
    mask[distance_squared <= outer_radius_voxels**2] += 1.0

    if z_cutoff is not None:
        z_mask = np.abs(z - center) <= z_cutoff
        mask *= z_mask
    return mask

def save_nifti_image(data, affine, filename):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)
    print(f"NIfTI file saved as '{filename}'")

def apply_deformation_field_to_image_itk(image_path, deformation_field_path, output_path, interp=sitk.sitkNearestNeighbor):
    """
    Apply a 3D deformation field to an image using SimpleITK.
    """
    image = sitk.ReadImage(image_path)
    deformation_field = sitk.ReadImage(deformation_field_path)
    if deformation_field.GetPixelID() != sitk.sitkVectorFloat64:
        deformation_field = sitk.Cast(deformation_field, sitk.sitkVectorFloat64)
    displacement_transform = sitk.DisplacementFieldTransform(deformation_field)
    resampled_image = sitk.Resample(
        image,
        displacement_transform,
        interp,
        0,
        image.GetPixelID()
    )
    sitk.WriteImage(resampled_image, output_path)
    print(f"Deformed image saved to: {output_path}")


def apply_deformation_field_to_image_scipy(image_path, deformation_field_path, output_path):
    image = nib.load(image_path)
    deformation_field = nib.load(deformation_field_path).get_fdata().astype(np.float64)

    deformation_field = np.permute_dims(deformation_field, (3, 0, 1, 2))  

    grid_x, grid_y, grid_z = np.indices(image.shape)
    grid_x = grid_x.astype(np.float64)
    grid_y = grid_y.astype(np.float64)
    grid_z = grid_z.astype(np.float64)
    #deformation_field = deformation_field.reshape(image.shape[0], image.shape[1], image.shape[2], 3)
    # Apply deformation field
    deformation_field = deformation_field + np.stack((grid_x, grid_y, grid_z), axis=0)
    #deformation_field = 
    deformed_image_data = map_coordinates(
        image.get_fdata(), deformation_field, order=1, mode='nearest'
    )
    #coords = np.indices(image.shape).reshape(3, -1).astype(np.float32)
    #coords += deformation_field.reshape(-1, 3).T
    #deformed_image_data = map_coordinates(image.get_fdata(), coords, order=1, mode='nearest')
    deformed_image = nib.Nifti1Image(deformed_image_data, image.affine)
    nib.save(deformed_image, output_path)
    print(f"Deformed image saved to: {output_path}")



def compute_affine_from_2pts(template_pts, target_pts):
    """
    Compute an affine transform (rotation+translation, no scaling) that maps two points in template to two points in target.
    """
    # Convert to numpy arrays
    t0, t1 = np.array(template_pts[0]), np.array(template_pts[1])
    s0, s1 = np.array(target_pts[0]), np.array(target_pts[1])

    # Compute direction vectors
    v_template = t1 - t0
    v_target = s1 - s0

    # Normalize
    v_template_norm = v_template / np.linalg.norm(v_template)
    v_target_norm = v_target / np.linalg.norm(v_target)

    # Compute rotation axis and angle
    axis = np.cross(v_template_norm, v_target_norm)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # Vectors are parallel, no rotation needed
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v_template_norm, v_target_norm), -1.0, 1.0))
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Compute translation
    t0_rot = R @ t0
    translation = s0 - t0_rot

    # netative of z component
    translation[2] *= -1  # Invert Z component to match ITK conventions

    # Build affine matrix
    affine1 = np.eye(4)
    #affine[:3, :3] = R
    affine1[:3, 3] = translation
    
    affine2 = np.eye(4)
    affine2[:3, :3] = R

    affine = affine2 @ affine1  # Combine rotation and translation
    return affine

def apply_affine_to_image(template_img_path, affine, output_path):
    img = sitk.ReadImage(template_img_path)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine[:3, :3].flatten())
    #transform.SetCenter(np.array(img.GetSize()) / 2.0)  # Set center to image center
    #transform.SetCenter([0,0,0])  # Set center to image center
    transform.SetTranslation(affine[:3, 3])
    #resampled = sitk.Resample(img, img, transform.GetInverse(), sitk.sitkLinear, 0.0)
    resampled = sitk.Resample(img, img, transform, sitk.sitkLinear, 0.0)

    sitk.WriteImage(resampled, output_path)
    print(f"Deformed image saved to: {output_path}")

# apply transform to itk deformation field
def apply_affine_to_deformation_field(deformation_field_path, affine, output_path):
    deformation_field = sitk.ReadImage(deformation_field_path)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine[:3, :3].flatten())
    transform.SetTranslation(affine[:3, 3])

    # Resample the deformation field
    resampled_field = sitk.Resample(deformation_field, deformation_field, transform, sitk.sitkLinear, 0.0)

    # Apply rotation to the displacement vectors
    arr = sitk.GetArrayFromImage(resampled_field)  # shape: (z, y, x, 3)
    R = affine[:3, :3]
    arr_shape = arr.shape
    arr_reshaped = arr.reshape(-1, 3)
    arr_rotated = (R @ arr_reshaped.T).T
    arr_rotated = arr_rotated.reshape(arr_shape)
    rotated_field = sitk.GetImageFromArray(arr_rotated, isVector=True)
    rotated_field.CopyInformation(resampled_field)

    sitk.WriteImage(rotated_field, output_path)
    print(f"Deformation field saved to: {output_path}")

