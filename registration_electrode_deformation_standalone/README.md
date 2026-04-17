# Registration Standalone Package

This directory contains a standalone copy of the registration and deformation code, plus the local helper modules it depends on.

The current workflow is script-based:
- `main_make_cylendrical_deformation_field.py` builds the deformation-field assets used by the deformation utilities
- `main_batch_register.py` runs atlas-to-subject registration for the configured subject list

## Contents

### Main Scripts
- `main_batch_register.py` - Batch registration entry point for the current multi-subject workflow
- `main_make_cylendrical_deformation_field.py` - Script for creating cylindrical deformation fields

### Dependencies (copied from rodreg and foreign_object)
- `utils.py` - General utility functions from rodreg
- `aligner.py` - Image alignment utilities
- `warp_utils.py` - Warping utilities
- `warper.py` - Main warping class
- `deform_losses.py` - Deformation loss functions
- `networks.py` - Neural network definitions
- `deform_image_by_electrode.py` - Electrode deformation functions
- `foreign_object_utils.py` - Utilities from foreign_object (renamed to avoid conflicts)

## Setup

1. Activate your Python environment:
```bash
source /home/ajoshi/my_venv/bin/activate
```

2. Install dependencies if needed:
```bash
pip install -r requirements.txt
```

## Configuration

`main_batch_register.py` currently uses hard-coded paths for:
- `RodentToolsDir`
- `RAW_BASE`
- `OUT_BASE`
- the Waxholm atlas image and label files

Before running the batch script, update those constants to match your local filesystem if your data is stored somewhere else.

## Usage

### Step 1: Generate Cylindrical Deformation Field
Before running the batch registration workflow, you must first generate the cylindrical deformation field and mask files:
```bash
python main_make_cylendrical_deformation_field.py
```

This produces local `.nii.gz` assets in this directory, including:
- `deformation_field_0.1mm.nii.gz`
- `deformation_field_0.1mm_itk.nii.gz`
- `cylinder_mask_0.1mm.nii.gz`

### Step 2: Run the Batch Registration Script
Once the deformation field and mask files are created, you can run the batch workflow:
```bash
python main_batch_register.py
```

The batch script currently processes:
- `r52`
- `r53`
- `r54`
- `r57`
- `r59`
- `r60`

## Notes

- All import paths have been updated to work with the local modules
- The batch script replaces the older subject-specific notebooks
- All core dependencies are included in this directory
- Registration inputs and outputs are controlled directly in `main_batch_register.py`

## Original Sources

- Main scripts from `/home/ajoshi/Projects/rodregdev/`
- Core modules from `/home/ajoshi/Projects/rodreg/`
- Foreign object utilities from `/home/ajoshi/Projects/rodregdev/foreign_object/`
