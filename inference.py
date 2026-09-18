"""
Example script to prepare algorithm submission to the PRISM-AI Challenge on GrandChallenge.

This script demonstrates how to:
  1. Read stacked .mha files (one per view, with multiple participants stacked
     along the first axis) and convert each 2D slice to a .dcm file, using the
     DICOM header fields stored in the corresponding JSON files.
  2. Load model weights / configuration from the resource path.
  3. Run a simple torch model on GPU (if available) for each participant in a stack.
  4. Write per-participant predictions (Year 1–5 breast cancer risk) as JSON.

The .mha → .dcm conversion produces standard DICOM files that can be read with
``pydicom.dcmread(path)``, which is the input format most algorithms expect.
The dummy model should be replaced with your own architecture and weights, but
you can re-use the handling of input and output.

This example is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

import glob
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import pydicom.uid
import SimpleITK as sitk
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH    = Path("/input")
OUTPUT_PATH   = Path("/output")
MODEL_DIR     = Path("/opt/ml/model")
# Resources baked into the container image, next to this script. Used as a
# fallback when no Model is attached to the algorithm (see load_model).
RESOURCE_PATH = Path(__file__).resolve().parent / "resources"

# Image sockets are mounted as subfolders of /input/images/<socket-slug>/,
# but JSON sockets are written as flat files directly in /input/<socket-slug>.json
# (no subfolder, and one single file per socket for the whole job).
IMAGES_PATH  = INPUT_PATH / "images"

# Tags that are written to file_meta rather than the main dataset
_FILE_META_TAGS = {"TransferSyntaxUID"}

# Tags we skip entirely (internal bookkeeping, not valid DICOM keywords)
_SKIP_TAGS = {"source_file"}

# Sequence tags whose items use keyword→value dicts (from our pipeline)
_SEQUENCE_TAGS = {"VOILUTSequence", "PerformedProtocolSequence",
                  "ViewCodeSequence", "ViewModifier"}


# ===================================================================
# 1.  Run model
# ===================================================================

# Mapping from view label to (images subfolder, metadata JSON filename)
# The JSON files sit directly in /input, they do NOT have their own subfolder,
# and their socket slug differs from the image slug ("...-metadata-stacked").
VIEW_DIRS = {
    "CC_L":  ("left-breast-cc-view-stacked",  "left-breast-cc-view-metadata-stacked.json"),
    "CC_R":  ("right-breast-cc-view-stacked", "right-breast-cc-view-metadata-stacked.json"),
    "MLO_L": ("left-breast-mlo-view-stacked", "left-breast-mlo-view-metadata-stacked.json"),
    "MLO_R": ("right-breast-mlo-view-stacked","right-breast-mlo-view-metadata-stacked.json"),
}


def run():
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model -------------------------------------------------------
    model = load_model(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Discover stacks by listing each view folder ----------------------
    # Grand Challenge assigns its own filename to every uploaded image (usually
    # a UUID), so a filename can NOT be reconstructed from the view label.
    # List the .mha files present in each view folder instead and match them
    # by position: every socket delivers the same stacks in the same order.
    mha_files = {}
    for view, (img_subdir, _) in VIEW_DIRS.items():
        view_dir = IMAGES_PATH / img_subdir
        mha_files[view] = sorted(view_dir.glob("*.mha"))
        if not mha_files[view]:
            raise FileNotFoundError(f"No .mha files found in {view_dir}")
        print(f"  {view:5s} -> {[f.name for f in mha_files[view]]}")

    n_stacks = len(mha_files["CC_L"])
    print(f"Found {n_stacks} stack(s)")

    # There is exactly one metadata JSON per socket per job, so a job must not
    # contain more than one stack — otherwise headers could not be matched.
    if n_stacks > 1:
        raise ValueError(
            f"Expected a single stack per job, found {n_stacks}. The metadata "
            f"JSON sockets provide only one file per view, so multiple stacks "
            f"cannot be matched to headers."
        )
    for view, files in mha_files.items():
        if len(files) != n_stacks:
            raise ValueError(
                f"{view} has {len(files)} .mha file(s) but CC_L has {n_stacks}"
            )

    # --- Create a temporary directory for .dcm files ----------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="gc_dicom_", dir="/tmp"))
    print(f"Writing temporary DICOM files to {tmp_dir}")

    try:
        all_predictions = []
        # Additional output, stacked per participant in the same order as
        # all_predictions. Left empty here, so an empty array is written.
        all_supplementary_output = []
        participant_counter = 0

        for stack_idx in range(n_stacks):
            stack_label = f"stack {stack_idx + 1}"
            print(f"\nProcessing {stack_label} …")

            # Load matched MHA + JSON for all four views
            stacks  = {}   # view → np.ndarray (N, H, W)
            spacings = {}  # view → (sx, sy)
            headers = {}   # view → list[dict]

            for view, (img_subdir, hdr_filename) in VIEW_DIRS.items():
                mha_path  = mha_files[view][stack_idx]
                json_path = INPUT_PATH / hdr_filename

                if not mha_path.exists():
                    raise FileNotFoundError(f"Missing MHA:  {mha_path}")
                if not json_path.exists():
                    raise FileNotFoundError(f"Missing JSON: {json_path}")

                stacks[view], spacings[view] = load_mha_stack(mha_path)
                headers[view]               = load_json_headers(json_path)

            # Sanity check: all four views must have the same number of slices
            n_participants = stacks["CC_L"].shape[0]
            for view in VIEW_DIRS:
                n = stacks[view].shape[0]
                n_hdr = len(headers[view])
                if n != n_participants:
                    raise ValueError(
                        f"{stack_label}: {view} MHA has {n} slices but "
                        f"CC_L has {n_participants}"
                    )
                if n_hdr != n_participants:
                    raise ValueError(
                        f"{stack_label}: {view} JSON has {n_hdr} records but "
                        f"MHA has {n_participants} slices"
                    )

            print(f"  {n_participants} participant(s) in this stack")

            for idx in range(n_participants):
                participant_counter += 1
                patient_id  = f"PARTICIPANT_{participant_counter:04d}"
                patient_dir = tmp_dir / patient_id
                study_uid   = pydicom.uid.generate_uid()

                dcm_lcc_path  = write_dcm_from_record(
                    pixel_array = stacks["CC_L"][idx],
                    output_path = patient_dir / "L_CC.dcm",
                    record      = headers["CC_L"][idx],
                    spacing     = spacings["CC_L"],
                    study_uid   = study_uid,
                )
                dcm_rcc_path  = write_dcm_from_record(
                    pixel_array = stacks["CC_R"][idx],
                    output_path = patient_dir / "R_CC.dcm",
                    record      = headers["CC_R"][idx],
                    spacing     = spacings["CC_R"],
                    study_uid   = study_uid,
                )
                dcm_lmlo_path = write_dcm_from_record(
                    pixel_array = stacks["MLO_L"][idx],
                    output_path = patient_dir / "L_MLO.dcm",
                    record      = headers["MLO_L"][idx],
                    spacing     = spacings["MLO_L"],
                    study_uid   = study_uid,
                )
                dcm_rmlo_path = write_dcm_from_record(
                    pixel_array = stacks["MLO_R"][idx],
                    output_path = patient_dir / "R_MLO.dcm",
                    record      = headers["MLO_R"][idx],
                    spacing     = spacings["MLO_R"],
                    study_uid   = study_uid,
                )

                # =============================================================
                # Plug in your own pipeline here
                #
                # At this point you have four standard .dcm files on disk:
                #   dcm_lcc_path  ->  patient_dir / "L_CC.dcm"
                #   dcm_rcc_path  ->  patient_dir / "R_CC.dcm"
                #   dcm_lmlo_path ->  patient_dir / "L_MLO.dcm"
                #   dcm_rmlo_path ->  patient_dir / "R_MLO.dcm"
                #
                # Your algorithm can read them with pydicom.dcmread(path) or
                # any other DICOM library.
                # =============================================================

                features = torch.tensor([[
                    extract_features_from_dicom(dcm_lcc_path),
                    extract_features_from_dicom(dcm_rcc_path),
                    extract_features_from_dicom(dcm_lmlo_path),
                    extract_features_from_dicom(dcm_rmlo_path),
                ]], dtype=torch.float32, device=device)

                with torch.no_grad():
                    preds = model(features)

                all_predictions.append(preds)
                print(f"  Participant {participant_counter}: {preds}")

                # Any additional output for this participant can be added here,
                # e.g. all_supplementary_output.append({"description": "..."}).
                # Add one entry per participant, so both output files line up.

        # --- Write output --------------------------------------------------
        write_json_file(
            location=OUTPUT_PATH / "breast-cancer-development-likelihood-stacked.json",
            content=all_predictions,
        )
        write_json_file(
            location=OUTPUT_PATH / "stacked-supplementary-output.json",
            content=all_supplementary_output,
        )
        print(f"\nPredictions saved for {participant_counter} participant(s).")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory {tmp_dir}")

    return 0


# ===================================================================
# 2.  Functions to handle input and output
# ===================================================================

def load_mha_stack(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Load a stacked .mha file and return (array, spacing).

    Parameters
    ----------
    path : Path
        Direct path to the .mha file.

    Returns
    -------
    array : np.ndarray, shape (N, H, W)
        One slice per participant.
    spacing : tuple[float, float]
        (x, y) pixel spacing in mm.
    """
    sitk_image = sitk.ReadImage(str(path))
    array      = sitk.GetArrayFromImage(sitk_image)   # shape: (N, H, W)
    spacing    = sitk_image.GetSpacing()[:2]           # (x, y) in mm
    return array, spacing


def load_json_headers(path: Path) -> list[dict]:
    """Load a JSON header file and return a list of per-participant dicts.

    Parameters
    ----------
    path : Path
        Direct path to the .json file.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(records)}")

    return records


def _format_age_string(value) -> str:
    """Convert an age to the DICOM 'AS' (Age String) format, e.g. 42 -> '042Y'.

    PatientAge is stored as a plain integer (years) in the metadata JSON, but
    VR 'AS' requires exactly four characters: three digits plus a unit
    (D=days, W=weeks, M=months, Y=years). Values that are already valid age
    strings are passed through unchanged.
    """
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 4 and text[:3].isdigit() and text[3] in "DWMY":
            return text
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            raise ValueError(f"no digits in age {value!r}")
        unit = text[-1].upper() if text[-1:].upper() in ("D", "W", "M", "Y") else "Y"
        return f"{int(digits):03d}{unit}"
    return f"{int(value):03d}Y"


def _build_sequence(items: list[dict]) -> pydicom.Sequence:
    """Reconstruct a pydicom Sequence from a list of keyword→value dicts."""
    seq = pydicom.Sequence()
    for item_dict in items:
        item_ds = pydicom.Dataset()
        for kw, val in item_dict.items():
            if val is None:
                continue
            try:
                setattr(item_ds, kw, val)
            except Exception as exc:
                print(f"    Warning: sequence sub-tag {kw!r} = {val!r}: {exc}")
        seq.append(item_ds)
    return seq


def write_dcm_from_record(
    pixel_array: np.ndarray,
    output_path: Path,
    record: dict,
    spacing: tuple[float, float],
    study_uid: str,
) -> Path:
    """Write a single 2-D numpy array to a DICOM file, applying all header
    fields stored in *record* (from the JSON produced by prism_pipeline.py).

    All tags are taken directly from the JSON record. The only additions
    outside the JSON are:
      - PixelData          : the actual pixel bytes from the array
      - HighBit            : derived as BitsStored - 1
      - SOPInstanceUID     : freshly generated per file for DICOM validity
      - StudyInstanceUID   : shared across all four views of one participant
      - SeriesInstanceUID  : freshly generated (required for valid DICOM,
                             not stored in JSON)
      - file_meta fields   : MediaStorageSOPClassUID / SOPInstanceUID

    Parameters
    ----------
    pixel_array : np.ndarray
        2-D array (H, W) with integer pixel values.
    output_path : Path
        Where to write the .dcm file.
    record : dict
        Header record for this participant and view, loaded from the JSON file.
    spacing : tuple[float, float]
        (x, y) pixel spacing in mm, taken from the MHA metadata.
        Note: PixelSpacing is not stored in the JSON. If your algorithms
        require it, consider adding tag (0028,0030) to the pipeline's TAG_MAP.
    study_uid : str
        Shared StudyInstanceUID for all four views of this participant.
    """
    array = pixel_array.astype(np.uint16)

    # ── File meta ──────────────────────────────────────────────────────────────
    sop_class = record.get("SOPClassUID") or "1.2.840.10008.5.1.4.1.1.1.2"
    sop_inst  = pydicom.uid.generate_uid()

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID    = sop_class
    file_meta.MediaStorageSOPInstanceUID = sop_inst
    # Always use uncompressed transfer syntax — pixel data from MHA is raw bytes.
    # The original TransferSyntaxUID in the JSON may be a compressed syntax
    # (e.g. JPEG Lossless), which would require encapsulation we don't do here.
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # ── Main dataset: apply all JSON fields ───────────────────────────────────
    ds             = pydicom.Dataset()
    ds.file_meta   = file_meta
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    for key, value in record.items():
        if value is None:
            continue
        if key in _SKIP_TAGS | _FILE_META_TAGS:
            continue

        if key == "PatientAge":
            try:
                value = _format_age_string(value)
            except Exception as exc:
                print(f"    Warning: could not format PatientAge {value!r}: {exc}")
                continue

        if key in _SEQUENCE_TAGS:
            if isinstance(value, list) and value:
                try:
                    setattr(ds, key, _build_sequence(value))
                except Exception as exc:
                    print(f"    Warning: could not set sequence {key!r}: {exc}")
            continue

        try:
            setattr(ds, key, value)
        except Exception as exc:
            print(f"    Warning: could not set {key!r} = {value!r}: {exc}")

    # ── Minimal additions outside the JSON ────────────────────────────────────
    ds.SOPInstanceUID   = sop_inst           # fresh, matches file_meta
    ds.StudyInstanceUID = study_uid          # shared across all four views
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.HighBit          = ds.BitsStored - 1  # always derivable from BitsStored
    ds.PixelData        = array.tobytes()

    # ── Write ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(output_path), ds, write_like_original=False)

    return output_path


def load_json_file(*, location: Path):
    with open(location) as f:
        return json.load(f)


def write_json_file(*, location: Path, content):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(content, f, indent=4)


# ===================================================================
# 3.  Dummy model
# ===================================================================

class BreastCancerRiskModel(nn.Module):
    """Trivially simple risk model for demonstration purposes.

    Architecture
    ------------
    For each of the four mammogram views the model computes a single scalar
    feature (global average pixel intensity, normalised to [0, 1]).  These
    four features are concatenated and passed through a small linear layer
    whose output is combined with year-specific baseline hazards to produce
    cumulative risk estimates for years 1–5.

    Weights and baseline hazards are loaded from ``config.json``.
    """

    def __init__(self, linear_weights: torch.Tensor, linear_bias: torch.Tensor,
                 baseline_hazards: list[float]):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(linear_weights)
            self.linear.bias.copy_(linear_bias)
        self.baseline_hazards = baseline_hazards

    def forward(self, view_features: torch.Tensor) -> dict[str, float]:
        """
        Parameters
        ----------
        view_features : torch.Tensor
            Shape (1, 4) – normalised mean intensities of the four views.

        Returns
        -------
        dict mapping ``"Year 1"`` … ``"Year 5"`` to cumulative risk.
        """
        risk_score = self.linear(view_features).item()

        survival = 1.0
        predictions = {}
        for year_idx, bh in enumerate(self.baseline_hazards, start=1):
            hazard = torch.sigmoid(torch.tensor(bh + risk_score)).item()
            survival *= (1.0 - hazard)
            predictions[f"Year {year_idx}"] = round(1.0 - survival, 6)

        return predictions


def load_model(device: torch.device) -> BreastCancerRiskModel:
    """Load model configuration and weights from the model path.
    In this example, we load weights, bias and baseline hazards. Your model
    configurations/weights may have a different format.
    """
    # An attached Model wins, so weights can be updated without rebuilding the
    # image; otherwise fall back to the copy baked into the image.
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        config_path = RESOURCE_PATH / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json at {MODEL_DIR} or {RESOURCE_PATH}. Either upload "
            f"model.tar.gz as a separate Model on the algorithm (see do_save.sh), "
            f"or make sure the Dockerfile copies config.json into the image."
        )
    print(f"Loading model configuration from {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    model = BreastCancerRiskModel(
        linear_weights  = torch.tensor([cfg["linear_weights"]], dtype=torch.float32),
        linear_bias     = torch.tensor(cfg["linear_bias"],      dtype=torch.float32),
        baseline_hazards= cfg["baseline_hazards"],
    )
    return model.to(device)


# ===================================================================
# 4.  Feature extraction helper
# ===================================================================

def extract_features_from_dicom(dcm_path: Path) -> float:
    """Read a DICOM file and extract a single scalar feature.

    In our demo algorithm the feature is simply the mean pixel intensity
    normalised to [0, 1].
    """
    dcm    = pydicom.dcmread(str(dcm_path))
    pixels = dcm.pixel_array
    return float(pixels.mean()) / 65535.0


# ===================================================================
# 5.  Utilities
# ===================================================================

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
