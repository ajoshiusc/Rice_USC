import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import NearestNDInterpolator


def multires_registration(fixed_image, moving_image, initial_transform):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(
        registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(
        registration_method.GetOptimizerStopConditionDescription()))
    return (final_transform, registration_method.GetMetricValue())


def interpolate_zeros(image_data, mask_data):
    # Find zero values in the image
    zero_indices = np.argwhere((image_data == 0) & (mask_data > 0))

    non_zero_indices = np.argwhere((image_data != 0) & (mask_data > 0))
    non_zero_values = image_data[non_zero_indices[:, 0],
                                 non_zero_indices[:, 1], non_zero_indices[:, 2]]

    # Create a NearestNDInterpolator object with non-zero voxel coordinates and values
    interpolator = NearestNDInterpolator(non_zero_indices, non_zero_values)

    # Create an array of voxel coordinates for zero indices
    zero_coords = zero_indices[:, [0, 1, 2]]

    # Use the interpolator to find the nearest non-zero values for all zero voxels
    interpolated_values = interpolator(zero_coords)

    # Create an array of voxel values for zero indices
    interpolated_data = image_data.copy()
    interpolated_data[zero_coords[:, 0], zero_coords[:, 1],
                      zero_coords[:, 2]] = interpolated_values

    return interpolated_data


def pad_nifti_image(input_path, output_path, padding_mm=10):
    # Load the NIfTI image using nibabel
    nifti_img = nib.load(input_path)
    img_data = nifti_img.get_fdata()
    original_dtype = img_data.dtype  # Store the original data type
    affine = nifti_img.affine

    # Get voxel sizes from the image header
    voxel_sizes = nifti_img.header.get_zooms()[:3]

    # Calculate the number of voxels to pad based on the given padding_mm
    padding_voxels = np.ceil(np.array(padding_mm) /
                             np.array(voxel_sizes)).astype(int)

    # Pad the image data with zeros
    padded_data = np.pad(img_data, ((padding_voxels[0], padding_voxels[0]),
                                    (padding_voxels[1], padding_voxels[1]),
                                    (padding_voxels[2], padding_voxels[2])),
                         mode='constant')

    # Update the affine matrix to reflect the new dimensions
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, 3] -= padding_voxels[i] * voxel_sizes[i]

    # Create a new nibabel NIfTI image with padded data and updated affine
    padded_nifti_img = nib.Nifti1Image(padded_data, new_affine)

    # Save the padded nibabel NIfTI image
    nib.save(padded_nifti_img, output_path)


if __name__ == "__main__":
    input_path = "input_image.nii.gz"  # Update with the actual input image path
    output_path = "padded_image.nii.gz"  # Update with the desired output path
    padding_mm = (10, 10, 10)  # Desired padding in millimeters (x, y, z)

    pad_nifti_image(input_path, output_path, padding_mm)
    print("Image padded with zeros and saved successfully.")
