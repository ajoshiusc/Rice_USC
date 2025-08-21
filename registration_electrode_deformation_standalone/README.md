# R57 Standalone Package

This directory contains a **fully tested and standalone** version of the R57 registration and deformation scripts, along with all their dependencies.

## ✅ Status: Ready to Use

All modules import successfully and the package is ready for production use.

## Contents

### Main Scripts
- `main_reg_for_Seymour_Ryan_R57.ipynb` - Main Jupyter notebook for R57 registration
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

### Testing
- `test_standalone.py` - Comprehensive test suite to verify functionality

## Setup

1. Activate your Python environment:
```bash
source /home/ajoshi/my_venv/bin/activate
```

2. (Optional) Install additional dependencies if needed:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
python test_standalone.py
```

### Step 1: Generate Cylindrical Deformation Field
Before running the Jupyter notebook, you must first generate the cylindrical deformation field and mask files:
```bash
python main_make_cylendrical_deformation_field.py
```

### Step 2: Run the Jupyter Notebook
Once the deformation field and mask files are created, you can proceed to run the main notebook:
```bash
jupyter lab main_reg_for_Seymour_Ryan_R57.ipynb
```

## ✅ Verification

The package has been tested and all modules import successfully:
- ✅ All external dependencies available
- ✅ All local modules import correctly  
- ✅ Main functions accessible
- ✅ Jupyter notebook readable

## Notes

- All import paths have been updated to work with the local modules
- The notebook no longer depends on external path additions
- All core dependencies are included in this directory
- Data files are expected to be in the paths specified in the scripts (typically `/deneb_disk/RodentTools/...`)

## Original Sources

- Main scripts from `/home/ajoshi/Projects/rodregdev/`
- Core modules from `/home/ajoshi/Projects/rodreg/`
- Foreign object utilities from `/home/ajoshi/Projects/rodregdev/foreign_object/`
