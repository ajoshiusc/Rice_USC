from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import SimpleITK as sitk

from deform_image_by_electrode import deform_image_by_electrode


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_CYL_PATH = MODULE_DIR / "cylinder_mask_0.1mm.nii.gz"
DEFAULT_TEMPLATE_DEFORMATION_FIELD_PATH = MODULE_DIR / "deformation_field_0.1mm_itk.nii.gz"
DEFAULT_TEMPLATE_PTS = [[0, 0, 0], [0, 0, -8]]


@dataclass(frozen=True)
class ElectrodeTrajectory:
    subject: str
    device: str
    tip_ras_mm: np.ndarray
    shaft_ras_mm: np.ndarray
    diameter_mm: float


def _read_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared_strings: List[str] = []
    for item in root.findall("a:si", XLSX_NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", XLSX_NS))
        shared_strings.append(text)
    return shared_strings


def _read_sheet_rows(xlsx_path: Path) -> List[List[str]]:
    with ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: List[List[str]] = []
    for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
        row_values: List[str] = []
        for cell in row.findall("a:c", XLSX_NS):
            value_node = cell.find("a:v", XLSX_NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            row_values.append(value)
        rows.append(row_values)
    return rows


def load_electrode_trajectories(xlsx_path: str | Path) -> Dict[str, List[ElectrodeTrajectory]]:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Electrode trajectory workbook not found: {xlsx_path}")

    rows = _read_sheet_rows(xlsx_path)
    if len(rows) < 3:
        raise ValueError(f"Workbook {xlsx_path} does not contain any trajectory rows.")

    trajectories: Dict[str, List[ElectrodeTrajectory]] = {}
    for row in rows[2:]:
        if len(row) < 9 or not row[0]:
            continue

        subject = row[0].strip()
        device = row[1].strip()
        tip = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64)
        shaft = np.array([float(row[5]), float(row[6]), float(row[7])], dtype=np.float64)
        diameter_mm = float(row[8])

        trajectories.setdefault(subject, []).append(
            ElectrodeTrajectory(
                subject=subject,
                device=device,
                tip_ras_mm=tip,
                shaft_ras_mm=shaft,
                diameter_mm=diameter_mm,
            )
        )

    return trajectories


def _target_pts_for_electrode(electrode: ElectrodeTrajectory) -> list[list[float]]:
    # Match the original foreign_object workflow: first point is the shaft point,
    # second point is the tip point.
    return [
        electrode.shaft_ras_mm.astype(float).tolist(),
        electrode.tip_ras_mm.astype(float).tolist(),
    ]


def _electrode_core_mask_from_template(
    deformed_template_path: Path,
    reference_image: sitk.Image,
) -> sitk.Image:
    deformed_template = sitk.ReadImage(str(deformed_template_path), sitk.sitkFloat32)
    core_mask = sitk.Cast(deformed_template > 1.5, sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return sitk.Cast(resampler.Execute(core_mask), sitk.sitkUInt8)


def deform_labels_by_electrodes(
    labels_path: str | Path,
    output_path: str | Path,
    electrodes: Iterable[ElectrodeTrajectory],
    influence_margin_mm: float = 0.6,
    expansion: float = 1.0,
    electrode_mask_output_path: str | Path | None = None,
    template_cyl_path: str | Path = DEFAULT_TEMPLATE_CYL_PATH,
    template_deformation_field_path: str | Path = DEFAULT_TEMPLATE_DEFORMATION_FIELD_PATH,
    template_pts: Sequence[Sequence[float]] = DEFAULT_TEMPLATE_PTS,
) -> str:
    del influence_margin_mm
    del expansion

    labels_path = Path(labels_path)
    output_path = Path(output_path)
    template_cyl_path = Path(template_cyl_path)
    template_deformation_field_path = Path(template_deformation_field_path)

    if not template_cyl_path.exists():
        raise FileNotFoundError(f"Template cylinder mask not found: {template_cyl_path}")
    if not template_deformation_field_path.exists():
        raise FileNotFoundError(f"Template deformation field not found: {template_deformation_field_path}")

    electrodes = list(electrodes)
    if not electrodes:
        raise ValueError("No electrode trajectories were provided.")

    reference_image = sitk.ReadImage(str(labels_path), sitk.sitkFloat32)
    combined_mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
    combined_mask.CopyInformation(reference_image)

    with TemporaryDirectory(prefix="electrode_label_deform_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        current_input_path = labels_path

        for index, electrode in enumerate(electrodes):
            step_prefix = temp_dir / f"electrode_{index}_{electrode.device.lower()}"
            masked_path = step_prefix.with_suffix(".masked.nii.gz")
            deformed_template_path = step_prefix.with_suffix(".template.nii.gz")
            deformed_field_path = step_prefix.with_suffix(".field.nii.gz")

            if index == len(electrodes) - 1:
                current_output_path = output_path
            else:
                current_output_path = step_prefix.with_suffix(".deformed.nii.gz")

            deform_image_by_electrode(
                target_path=str(current_input_path),
                target_electrode_path=str(masked_path),
                target_electrode_deformed_path=str(current_output_path),
                template_cyl_path=str(template_cyl_path),
                template_deformation_field_path=str(template_deformation_field_path),
                template_pts=template_pts,
                target_pts=_target_pts_for_electrode(electrode),
                output_path=str(deformed_template_path),
                output_deformation_field_path=str(deformed_field_path),
                islabel=True,
            )

            combined_mask = sitk.Maximum(
                combined_mask,
                _electrode_core_mask_from_template(deformed_template_path, reference_image),
            )
            current_input_path = current_output_path

    if electrode_mask_output_path is not None:
        sitk.WriteImage(combined_mask, str(electrode_mask_output_path))

    return str(output_path)


def deform_subject_labels(
    subject_id: str,
    labels_path: str | Path,
    trajectories_by_subject: Dict[str, List[ElectrodeTrajectory]],
    output_path: str | Path,
    influence_margin_mm: float = 0.6,
    expansion: float = 1.0,
    electrode_mask_output_path: str | Path | None = None,
    template_cyl_path: str | Path = DEFAULT_TEMPLATE_CYL_PATH,
    template_deformation_field_path: str | Path = DEFAULT_TEMPLATE_DEFORMATION_FIELD_PATH,
    template_pts: Sequence[Sequence[float]] = DEFAULT_TEMPLATE_PTS,
) -> str:
    subject_electrodes = trajectories_by_subject.get(subject_id, [])
    if not subject_electrodes:
        raise KeyError(f"No electrode trajectories found for subject '{subject_id}'.")

    return deform_labels_by_electrodes(
        labels_path=labels_path,
        output_path=output_path,
        electrodes=subject_electrodes,
        influence_margin_mm=influence_margin_mm,
        expansion=expansion,
        electrode_mask_output_path=electrode_mask_output_path,
        template_cyl_path=template_cyl_path,
        template_deformation_field_path=template_deformation_field_path,
        template_pts=template_pts,
    )


def deform_subject_labels_from_workbook(
    subject_id: str,
    labels_path: str | Path,
    workbook_path: str | Path,
    output_path: str | Path,
    influence_margin_mm: float = 0.6,
    expansion: float = 1.0,
    electrode_mask_output_path: str | Path | None = None,
    template_cyl_path: str | Path = DEFAULT_TEMPLATE_CYL_PATH,
    template_deformation_field_path: str | Path = DEFAULT_TEMPLATE_DEFORMATION_FIELD_PATH,
    template_pts: Sequence[Sequence[float]] = DEFAULT_TEMPLATE_PTS,
) -> str:
    trajectories_by_subject = load_electrode_trajectories(workbook_path)
    return deform_subject_labels(
        subject_id=subject_id,
        labels_path=labels_path,
        trajectories_by_subject=trajectories_by_subject,
        output_path=output_path,
        influence_margin_mm=influence_margin_mm,
        expansion=expansion,
        electrode_mask_output_path=electrode_mask_output_path,
        template_cyl_path=template_cyl_path,
        template_deformation_field_path=template_deformation_field_path,
        template_pts=template_pts,
    )
