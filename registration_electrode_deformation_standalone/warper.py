#!/usr/bin/env python3

from monai.utils import set_determinism
from monai.networks.nets import GlobalNet, LocalNet, RegUNet, unet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp, DVF2DDF
import torch
from torch.nn import MSELoss
from monai.transforms import (
    LoadImage,
    Resize,
    EnsureChannelFirst,
    ScaleIntensityRangePercentiles,
)
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import (
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelationLoss,
)
from nilearn.image import resample_to_img, resample_img, crop_img, load_img
from torch.nn.functional import grid_sample
from warp_utils import get_grid, apply_warp, jacobian_determinant, jacobian_determinant_torch
from typing import List
from monai.losses import BendingEnergyLoss
from deform_losses import BendingEnergyLoss as myBendingEnergyLoss
from deform_losses import GradEnergyLoss
from networks import LocalNet2
import argparse
import nibabel as nib
import numpy as np


class dscolors:
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    purple = "\033[95m"
    cyan = "\033[96m"
    clear = "\033[0m"
    bold = "\033[1m"
    ul = "\033[4m"


class Warper:
    # device = 'cuda'
    # max_epochs = 3000
    # lr = .01
    def __init__(self):
        set_determinism(42)

    # def setLoss(self, loss):
    # 	self.loss=loss
    # 	if loss == 'mse':
    # 		image_loss = MSELoss()
    # 	elif loss == 'cc':
    # 		image_loss = LocalNormalizedCrossCorrelationLoss()
    # 	elif loss == 'mi':
    # 		image_loss = GlobalMutualInformationLoss()
    # 	else:
    # 		AssertionError

    # 	set_determinism(42)

    def loadMoving(self, moving_file):
        self.moving, self.moving_meta = LoadImage(image_only=False)(moving_file)
        self.moving = EnsureChannelFirst()(self.moving)

    def loadTarget(self, fixed_file):
        self.target, self.target_meta = LoadImage(image_only=False)(fixed_file)
        self.target = EnsureChannelFirst()(self.target)

    def loadTargetMask(self, target_mask):
        if target_mask is None:
            self.target_mask = None
        else:
            self.target_mask, self.target_mask_meta = LoadImage(image_only=False)(
                target_mask
            )
            self.target_mask = self.target_mask > 0.5
            self.target_mask.type(torch.DoubleTensor)
            self.target_mask = EnsureChannelFirst()(self.target_mask)

    def saveWarpedLabels(self, label_file, output_label_file):
        print(dscolors.green + "warping " + label_file + dscolors.clear)
        print(
            dscolors.green
            + "saving warped labels: "
            + dscolors.clear
            + output_label_file
            + dscolors.clear
        )
        label, meta = LoadImage(image_only=False)(label_file)
        label = EnsureChannelFirst()(label)
        warped_labels = apply_warp(
            self.ddf[None,], label[None,], self.target[None,], interp_mode="nearest"
        )
        # write_nifti(warped_labels[0, 0], output_label_file, affine=self.target.affine)
        nib.save(
            nib.Nifti1Image(
                warped_labels[0, 0].detach().cpu().numpy(), self.target.affine
            ),
            output_label_file,
        )

    def nonlinear_reg(
        self,
        target_file,
        moving_file,
        output_file,
        target_mask=None,
        label_file=None,
        ddf_file=None,
        inv_ddf_file=None,
        output_label_file=None,
        jacobian_determinant_file=None,
        inv_jacobian_determinant_file=None,
        loss="cc",
        reg_penalty=0.3,
        nn_input_size=64,
        lr=1e-6,
        max_epochs=1000,
        device="cuda",
        use_diffusion_reg=False,
    ):
        if loss == "mse":
            image_loss = MSELoss()
        elif loss == "cc":
            # Larger kernel for better subcortical structure capture
            image_loss = LocalNormalizedCrossCorrelationLoss(kernel_size=13)
        elif loss == "mi":
            image_loss = GlobalMutualInformationLoss()
        else:
            raise AssertionError("Invalid Loss")

        # Use diffusion (gradient) regularization for smoother local deformations
        if use_diffusion_reg:
            regularization = GradEnergyLoss()
        else:
            regularization = myBendingEnergyLoss()
        #######################
        set_determinism(42)
        self.loadMoving(moving_file)
        self.loadTarget(target_file)
        self.loadTargetMask(target_mask)

        SZ = nn_input_size
        moving_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(self.moving).to(
            device
        )
        target_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(self.target).to(
            device
        )

        if target_mask is not None:
            target_mask_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(
                self.target_mask
            ).to(device)

        # Improved normalization for cross-protocol T2 (atlas vs ex vivo)
        # Normalize to [0, 1] with robust percentiles
        moving_ds = ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        )(moving_ds)
        target_ds = ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        )(target_ds)
        
        # Apply histogram matching to reduce protocol differences
        # Match moving (atlas) to target (ex vivo) intensity distribution
        moving_flat = moving_ds.flatten().cpu().numpy()
        target_flat = target_ds.flatten().cpu().numpy()
        
        # Simple histogram matching using quantiles
        moving_sorted_idx = np.argsort(moving_flat)
        target_sorted = np.sort(target_flat)
        
        # Map moving intensities to target distribution
        moving_matched_flat = np.zeros_like(moving_flat)
        moving_matched_flat[moving_sorted_idx] = target_sorted[
            (np.arange(len(moving_flat)) * len(target_flat) // len(moving_flat)).clip(0, len(target_flat)-1)
        ]
        moving_ds = torch.from_numpy(moving_matched_flat.reshape(moving_ds.shape)).to(device).float()
        # Improved UNet with instance norm and wider channels
        # for better feature extraction of subcortical structures
        reg = unet.UNet(
            spatial_dims=3,  # spatial dims
            in_channels=2,
            out_channels=3,  # output channels (to represent 3D displacement vector field)
            channels=(32, 64, 128, 256, 320),  # Wider network for better feature extraction
            strides=(2, 2, 2, 2),  # convolutional strides
            norm="instance",  # Instance norm better for registration
            dropout=0.1,  # Light dropout for regularization
        ).to(device)
        if USE_COMPILED:
            warp_layer = Warp(3, padding_mode="zeros").to(device)
        else:
            warp_layer = Warp("bilinear", padding_mode="zeros").to(device)
        reg.train()
        # Use AdamW with weight decay for better generalization
        optimizerR = torch.optim.AdamW(reg.parameters(), lr=lr, weight_decay=1e-5)
        # Faster LR reduction for ex vivo registration
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizerR, mode='min', factor=0.5, patience=50, min_lr=1e-7
        )
        print(dscolors.green + "optimizing" + dscolors.clear)

        dvf_to_ddf = DVF2DDF()
        
        # Store original unmasked images for better gradient flow
        target_ds_orig = target_ds.clone()
        moving_ds_orig = moving_ds.clone()
        
        # Convert mask to soft weights (0.01 outside, 1.0 inside) for better gradient flow
        if target_mask is not None:
            mask_weight = target_mask_ds * 0.99 + 0.01
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 200

        for epoch in range(max_epochs):
            optimizerR.zero_grad()
            
            # Use original unmasked images as input for better feature learning
            input_data = torch.cat((moving_ds_orig, target_ds_orig), dim=0)
            input_data = input_data[None,]
            dvf_ds = reg(input_data)
            ddf_ds = dvf_to_ddf(dvf_ds)
            inv_ddf_ds = dvf_to_ddf(-dvf_ds)

            image_moved = warp_layer(moving_ds_orig[None,], ddf_ds)

            # Apply mask as soft weight in loss computation instead of hard multiplication
            if target_mask is not None:
                # Weighted loss: emphasize brain regions but allow learning outside
                imgloss = image_loss(image_moved * mask_weight, target_ds_orig[None,] * mask_weight)
            else:
                imgloss = image_loss(image_moved, target_ds_orig[None,])
            
            regloss = reg_penalty * regularization(ddf_ds)
            
            # Add penalty for folding (negative Jacobian determinant)
            jac_det = jacobian_determinant_torch(ddf_ds[0])
            folding_penalty = torch.sum(torch.relu(-jac_det)) * 0.1
            
            vol_loss = imgloss + regloss + folding_penalty

            vol_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(reg.parameters(), max_norm=1.0)
            optimizerR.step()
            scheduler.step(vol_loss)
            
            # Early stopping check
            if vol_loss.item() < best_loss:
                best_loss = vol_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            if epoch % 50 == 0:
                print(
                    f"\nepoch: {epoch}/{max_epochs}, Loss: {vol_loss.item():.4f}, "
                    f"ImgLoss: {imgloss.item():.4f}, RegLoss: {regloss.item():.4f}, "
                    f"Fold: {folding_penalty.item():.4f}, LR: {optimizerR.param_groups[0]['lr']:.2e}"
                )
            else:
                print(
                    "epoch:",
                    dscolors.green,
                    f"{epoch}/{max_epochs}",
                    "Loss:",
                    dscolors.yellow,
                    f"{vol_loss.detach().cpu().numpy():.4f}",
                    dscolors.clear,
                    "",
                    end="\r",
                )

        print("finished", dscolors.green, f"{max_epochs}", dscolors.clear, "epochs")

        print("\n\n")
        # write_nifti(image_moved[0, 0], 'moved_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(target_ds[0], 'target_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(moving_ds[0], 'moving_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(torch.permute(ddf_ds[0],[1,2,3,0]),'ddf_ds.nii.gz',affine=target_ds.affine)
        # jdet_ds = jacobian_determinant(ddf_ds[0])
        # write_nifti(jdet_ds,'jdet_ds.nii.gz',affine=target_ds.affine)

        print(dscolors.green + "computing deformation field" + dscolors.clear)
        size_moving = self.moving[0].shape
        size_target = self.target[0].shape
        ddfx = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 0].to('cpu')).to('cpu') * (
            size_moving[0] / SZ
        )
        ddfy = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 1]).to('cpu') * (
            size_moving[1] / SZ
        )
        ddfz = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 2].to('cpu')).to('cpu') * (
            size_moving[2] / SZ
        )
        self.ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
        del ddf_ds, ddfx, ddfy, ddfz

        print(dscolors.green + "computing inverse deformation field" + dscolors.clear)
        size_moving = self.moving[0].shape
        size_target = self.target[0].shape
        ddfx = Resize(spatial_size=size_moving, mode="trilinear")(inv_ddf_ds[:, 0].to('cpu')).to('cpu') * (
            size_target[0] / SZ
        )
        ddfy = Resize(spatial_size=size_moving, mode="trilinear")(inv_ddf_ds[:, 1].to('cpu')).to('cpu') * (
            size_target[1] / SZ
        )
        ddfz = Resize(spatial_size=size_moving, mode="trilinear")(inv_ddf_ds[:, 2].to('cpu')).to('cpu') * (
            size_target[2] / SZ
        )
        self.inv_ddf = torch.cat((ddfx, ddfy, ddfz), dim=0).to('cpu')
        del inv_ddf_ds, ddfx, ddfy, ddfz

        # Apply the warp
        print(dscolors.green + "applying warp" + dscolors.clear)
        image_movedo = apply_warp(
            self.ddf[None,], self.moving[None,], self.target[None,]
        )
        print(dscolors.green + "saving warped output: " + dscolors.clear + output_file)
        # write_nifti(image_movedo[0, 0], output_file, affine=self.target.affine)
        nib.save(
            nib.Nifti1Image(
                image_movedo[0, 0].detach().cpu().numpy(), self.target.affine
            ),
            output_file,
        )

        if ddf_file is not None:
            print(dscolors.green + "saving ddf: " + dscolors.clear + ddf_file)
            nib.save(
                nib.Nifti1Image(
                    torch.permute(self.ddf, [1, 2, 3, 0]).detach().cpu().numpy(),
                    self.target.affine,
                ),
                ddf_file,
            )

        if inv_ddf_file is not None:
            print(dscolors.green + "saving inv_ddf: " + dscolors.clear + inv_ddf_file)
            nib.save(
                nib.Nifti1Image(
                    torch.permute(self.inv_ddf, [1, 2, 3, 0]).detach().cpu().numpy(),
                    self.moving.affine,
                ),
                inv_ddf_file,
            )

        # Apply the warp to labels
        if label_file is not None and output_label_file is not None:
            print(dscolors.green + "warping " + label_file + dscolors.clear)
            print(
                dscolors.green
                + "saving warped labels: "
                + dscolors.clear
                + output_label_file
                + dscolors.clear
            )
            label, meta = LoadImage(image_only=False)(label_file)
            label = EnsureChannelFirst()(label)
            warped_labels = apply_warp(
                self.ddf[None,], label[None,], self.target[None,], interp_mode="nearest"
            )
            # write_nifti(warped_labels[0,0], output_label_file, affine=self.target.affine)
            nib.save(
                nib.Nifti1Image(
                    warped_labels[0, 0].detach().cpu().numpy(), self.target.affine
                ),
                output_label_file,
            )

        if jacobian_determinant_file is not None:
            jdet = jacobian_determinant(self.ddf)
            # write_nifti(jdet,'jdet.nii.gz',affine=self.target.affine)
            nib.save(
                nib.Nifti1Image(jdet, self.target.affine), jacobian_determinant_file
            )

        if inv_jacobian_determinant_file is not None:
            ijdet = jacobian_determinant(self.inv_ddf)
            # write_nifti(jdet,'jdet.nii.gz',affine=self.target.affine)
            nib.save(
                nib.Nifti1Image(ijdet, self.moving.affine),
                inv_jacobian_determinant_file,
            )


#####################
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Nonlinear registration for mouse brains"
    )
    parser.add_argument("moving_file", type=str, help="moving file name")
    parser.add_argument("fixed_file", type=str, help="fixed file name")
    parser.add_argument("output_file", type=str, help="output file name")
    parser.add_argument(
        "--label-file", "--label", type=str, help="input label file name"
    )
    parser.add_argument("--output-label-file", type=str, help="output label file name")
    parser.add_argument("-j", "--jacobian", type=str, help="output jacobian file name")
    parser.add_argument(
        "-ddf",
        "--ddf-file",
        type=str,
        default="",
        help="dense displacement field file name",
    )
    parser.add_argument(
        "-iddf",
        "--inv-ddf-file",
        type=str,
        default="",
        help="inverse dense displacement field file name",
    )

    parser.add_argument(
        "--nn_input_size",
        type=int,
        default=64,
        help="size of the neural network input (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "-e", "--max-epochs", type=int, default=3000, help="maximum interations"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device: cuda, cpu, etc."
    )
    parser.add_argument(
        "-l", "--loss", type=str, default="cc", help="loss function: mse, cc or mi"
    )
    parser.add_argument(
        "-r",
        "--reg-penalty",
        type=str,
        default=0.3,
        help="loss function: mse, cc or mi",
    )
    # parser.add_argument('--')

    args = parser.parse_args()
    warper = Warper()

    warper.nonlinear_reg(
        target_file=args.fixed_file,
        moving_file=args.moving_file,
        output_file=args.output_file,
        ddf_file=args.ddf_file,
        inv_ddf_file=args.inv_ddf_file,
        label_file=args.label_file,
        output_label_file=args.output_label_file,
        jacobian_determinant_file=args.jacobian,
        loss=args.loss,
        reg_penalty=args.reg_penalty,
        nn_input_size=args.nn_input_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
