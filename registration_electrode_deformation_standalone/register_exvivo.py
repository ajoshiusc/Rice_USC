from aligner import Aligner
import os
import argparse
import numpy as np
import nibabel as nb
import SimpleITK as sitk
from scipy import ndimage
import torch

# Import local modules
# Ensure these are in the python path or same directory
from warper import Warper
from utils import multires_registration

def create_atlas_guided_mask(subject_path, atlas_path, affine_transform_path, output_mask_path):
    """
    Creates a robust brain mask for the subject by:
    1. Warping the atlas mask to subject space (using affine transform).
    2. Intersecting with an intensity-based threshold mask of the subject.
    """
    print(f"Creating atlas-guided mask...")
    
    # 1. Load or Create Atlas Mask
    atlas_img = nb.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    # Simple threshold to get atlas brain mask if one doesn't exist
    atlas_mask_data = (atlas_data > np.percentile(atlas_data[atlas_data > 0], 5)).astype(np.uint8)
    
    # Save temp atlas mask
    temp_atlas_mask_path = "temp_atlas_mask.nii.gz"
    nb.save(nb.Nifti1Image(atlas_mask_data, atlas_img.affine, atlas_img.header), temp_atlas_mask_path)

    # 2. Apply Affine Transform to Atlas Mask
    fixed_image = sitk.ReadImage(subject_path, sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(temp_atlas_mask_path, sitk.sitkUInt8)
    
    affine_transform = sitk.ReadTransform(affine_transform_path)
    
    # Resample mask (Nearest Neighbor)
    warped_mask = sitk.Resample(moving_mask, fixed_image, affine_transform, sitk.sitkNearestNeighbor)
    
    # Convert to numpy
    warped_atlas_mask_data = sitk.GetArrayFromImage(warped_mask)

    # 3. Create Intensity Mask from Subject
    sub_img = nb.load(subject_path)
    sub_data = sub_img.get_fdata()
    # Threshold low intensity background/artifacts
    intensity_mask = (sub_data > np.percentile(sub_data[sub_data > 0], 5)).astype(np.uint8)

    # 4. Combine (Intersection)
    # Transpose sitk array to match nibabel if needed? 
    # SITK is (z, y, x), Nibabel is (x, y, z).
    # sitk.GetArrayFromImage returns (z, y, x).
    # sub_data is (x, y, z).
    # We need to be careful.
    # Better to use sitk for everything or nibabel for everything.
    # Let's use SITK for intensity mask too to be safe.
    
    fixed_arr = sitk.GetArrayFromImage(fixed_image)
    intensity_mask_sitk = (fixed_arr > np.percentile(fixed_arr[fixed_arr > 0], 5)).astype(np.uint8)
    
    combined_mask = (warped_atlas_mask_data * intensity_mask_sitk).astype(np.uint8)

    # 5. Morphological Cleanup
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    combined_mask = ndimage.binary_opening(combined_mask, iterations=2)
    combined_mask = ndimage.binary_dilation(combined_mask, iterations=3)
    
    # Cast to uint8 for SimpleITK
    combined_mask = combined_mask.astype(np.uint8)

    # Save using SITK to ensure geometry matches
    final_mask_sitk = sitk.GetImageFromArray(combined_mask)
    final_mask_sitk.CopyInformation(fixed_image)
    sitk.WriteImage(final_mask_sitk, output_mask_path)
    
    # Cleanup temps
    if os.path.exists(temp_atlas_mask_path): os.remove(temp_atlas_mask_path)
    
    print(f"Mask saved to {output_mask_path}")
    return output_mask_path

def run_registration(fixed_image_path, moving_atlas_path, atlas_labels_path, output_dir, skip_affine=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import shutil

    def strip_nii_gz(fname):
        base = os.path.basename(fname)
        for ext in ['.nii.gz', '.nii', '.img']:
            if base.endswith(ext):
                return base[: -len(ext)]
        return os.path.splitext(base)[0]

    fixed_base = strip_nii_gz(fixed_image_path)
    moving_base = strip_nii_gz(moving_atlas_path)
    labels_base = strip_nii_gz(atlas_labels_path)

    # Copy original files to output directory
    shutil.copy2(fixed_image_path, os.path.join(output_dir, f"{fixed_base}_orig.nii.gz"))
    shutil.copy2(moving_atlas_path, os.path.join(output_dir, f"{moving_base}_orig.nii.gz"))
    shutil.copy2(atlas_labels_path, os.path.join(output_dir, f"{labels_base}_orig.nii.gz"))

    # Use derived base for output prefix
    basename = os.path.join(output_dir, f"atlas_to_{fixed_base}_reg")
    
    # --- 1. Rigid Registration (SimpleITK) ---
    centered_atlas_path = basename + ".rigid.nii.gz"
    centered_labels_path = basename + ".rigid.label.nii.gz"
    rigid_transform_path = basename + ".rigid.tfm"

    fixed_sitk = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_sitk = sitk.ReadImage(moving_atlas_path, sitk.sitkFloat32)

    print("\n--- Step 1: Rigid Registration ---")
    labels_sitk = None
    if not os.path.exists(rigid_transform_path):
        # Initialize
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Register
        final_rigid_transform, _ = multires_registration(fixed_sitk, moving_sitk, initial_transform)
        sitk.WriteTransform(final_rigid_transform, rigid_transform_path)

        # Resample Image
        moved_image = sitk.Resample(moving_sitk, fixed_sitk, final_rigid_transform)
        sitk.WriteImage(moved_image, centered_atlas_path)

        # Resample Labels (Nearest Neighbor)
        labels_sitk = sitk.ReadImage(atlas_labels_path, sitk.sitkUInt16)
        moved_labels = sitk.Resample(labels_sitk, fixed_sitk, final_rigid_transform, sitk.sitkNearestNeighbor)
        sitk.WriteImage(moved_labels, centered_labels_path)
    else:
        print("Rigid transform found, skipping registration.")
        final_rigid_transform = sitk.ReadTransform(rigid_transform_path)

        # Ensure resampled outputs exist; if missing, create them using the stored transform
        if not os.path.exists(centered_atlas_path):
            moved_image = sitk.Resample(moving_sitk, fixed_sitk, final_rigid_transform)
            sitk.WriteImage(moved_image, centered_atlas_path)
        if not os.path.exists(centered_labels_path):
            labels_sitk = sitk.ReadImage(atlas_labels_path, sitk.sitkUInt16)
            moved_labels = sitk.Resample(labels_sitk, fixed_sitk, final_rigid_transform, sitk.sitkNearestNeighbor)
            sitk.WriteImage(moved_labels, centered_labels_path)
    # Always load labels_sitk before affine/skip_affine block
    if labels_sitk is None:
        labels_sitk = sitk.ReadImage(atlas_labels_path, sitk.sitkUInt16)



    # --- 2. Affine Registration (SimpleITK) ---
    affine_atlas_path = basename + ".affine.nii.gz"
    affine_labels_path = basename + ".affine.label.nii.gz"
    affine_ddf_path = basename + ".affine_ddf.nii.gz"


    if skip_affine:
        print("\n--- Step 2: Affine Registration SKIPPED (using rigid only) ---")
        # Use only rigid transform for subsequent steps
        moved_image_affine = sitk.Resample(moving_sitk, fixed_sitk, final_rigid_transform)
        sitk.WriteImage(moved_image_affine, affine_atlas_path)
        moved_labels_affine = sitk.Resample(labels_sitk, fixed_sitk, final_rigid_transform, sitk.sitkNearestNeighbor)
        sitk.WriteImage(moved_labels_affine, affine_labels_path)
    else:
        print("\n--- Step 2: Affine Registration (Deep Learning) ---")
        aligner = Aligner()
        aligner.affine_reg(
            fixed_image_path,
            centered_atlas_path,  # Use rigid-registered atlas as moving
            affine_atlas_path,
            affine_ddf_path,
            loss='mse',
            #nn_input_size=64,
            #lr=1e-4,
            max_epochs=5000,
            #device='cuda'
        )
        # Warp labels using the affine DDF
        # Use the same warping utility as in nonlinear_reg for labels
        from warp_utils import apply_warp
        import nibabel as nib
        
        # Load target to get correct reference shape
        target_img = nib.load(fixed_image_path)
        target_data = target_img.get_fdata()
        target_tensor = torch.from_numpy(target_data).unsqueeze(0).unsqueeze(0).float()
        
        # Load rigid-registered labels
        label_img = nib.load(centered_labels_path)
        label_data = label_img.get_fdata()
        label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0).float()
        
        # Load affine DDF
        affine_ddf = nib.load(affine_ddf_path).get_fdata()
        ddf_tensor = torch.from_numpy(np.moveaxis(affine_ddf, -1, 0)).unsqueeze(0).float()
        
        # Warp labels with target reference
        warped_labels = apply_warp(ddf_tensor, label_tensor, target_tensor, interp_mode="nearest")
        warped_labels_np = warped_labels[0, 0].detach().cpu().numpy()
        nib.save(nib.Nifti1Image(warped_labels_np, target_img.affine), affine_labels_path)


    # --- 3. Mask Generation ---
    print("\n--- Step 3: Mask Generation ---")
    mask_path = basename + ".mask.nii.gz"
    # Use affine-registered atlas directly for mask generation
    # Create simple intensity-based mask from affine-registered atlas and fixed image
    fixed_img = nb.load(fixed_image_path)
    fixed_data = fixed_img.get_fdata()
    affine_img = nb.load(affine_atlas_path)
    affine_data = affine_img.get_fdata()
    
    # Create masks from both images
    fixed_mask = (fixed_data > np.percentile(fixed_data[fixed_data > 0], 5)).astype(np.uint8)
    affine_mask = (affine_data > np.percentile(affine_data[affine_data > 0], 5)).astype(np.uint8)
    
    # Combine masks (intersection)
    combined_mask = (fixed_mask * affine_mask).astype(np.uint8)
    
    # Morphological cleanup
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    combined_mask = ndimage.binary_opening(combined_mask, iterations=2)
    combined_mask = ndimage.binary_dilation(combined_mask, iterations=3)
    combined_mask = combined_mask.astype(np.uint8)
    
    # Save mask
    nb.save(nb.Nifti1Image(combined_mask, fixed_img.affine), mask_path)
    print(f"Mask saved to {mask_path}")


    # --- 4. Nonlinear Registration (Monai/Warper) ---
    print("\n--- Step 4: Nonlinear Registration ---")
    final_atlas_path = basename + ".nonlin.nii.gz"
    final_labels_path = basename + ".nonlin.label.nii.gz"
    nonlin_ddf_path = basename + ".nonlin_ddf.nii.gz"
    
    warper = Warper()
    warper.nonlinear_reg(
        target_file=fixed_image_path,
        moving_file=affine_atlas_path,  # Use Affine registered image
        output_file=final_atlas_path,
        target_mask=mask_path,
        ddf_file=nonlin_ddf_path,
        label_file=affine_labels_path,  # Use Affine registered labels
        output_label_file=final_labels_path,
        
        # Optimized Parameters for Ex Vivo to Atlas Registration
        # Focus on hippocampal and subcortical alignment
        reg_penalty=0.15*(64**3/128**3),      # Reduced regularization for more flexible local deformations
        nn_input_size=128,    # Higher resolution to preserve hippocampal boundaries
        lr=2e-4,              # Higher initial LR with adaptive schedule
        max_epochs=75000,      # More epochs with early stopping
        loss="cc",            # Cross-correlation with larger kernel (13)
        use_diffusion_reg=True,  # Use gradient regularization for smoother deformations
        kernel_size=7,        # Smaller kernel for finer details
    )

    print("\n--- Registration Complete ---")
    print(f"Final Registered Image: {final_atlas_path}")
    print(f"Final Registered Labels: {final_labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Atlas Registration for Ex Vivo Rodent Brains")
    parser.add_argument("--fixed", required=True, help="Path to fixed (subject) skull-stripped T2 image")
    parser.add_argument("--atlas", required=True, help="Path to moving atlas T2 image")
    parser.add_argument("--labels", required=True, help="Path to atlas label map")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--skip_affine", action="store_true", help="Skip affine registration and use only rigid transform for all subsequent steps.")
    args = parser.parse_args()
    run_registration(args.fixed, args.atlas, args.labels, args.output, skip_affine=args.skip_affine)
