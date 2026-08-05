# 📦 PRISM-AI Example Scripts

## Overview

This repository contains example scripts to help participants prepare their algorithm for submission to the **PRISM-AI Challenge** on [Grand Challenge](https://grand-challenge.org/).

In personalized breast cancer screening, accurate risk estimates could help tailor screening intensity, improving early detection for some while reducing unnecessary procedures for others. AI-driven risk assessment could be the next major step forward. Several commercial and academic AI algorithms now predict future breast cancer risk from screening mammograms. The PRISM-AI Challenge provides a transparent, independent, and rigorous evaluation of AI algorithms for breast cancer risk prediction on a large population-based screening cohort.

------------------------------------------------------------------------

## 📁 Repository Contents

### 🦿 `inference.py`

The main inference script for running a basic algorithm on the Grand Challenge platform.

-   Mammograms are provided as stacked `.mha` files, with separate stacks for each of the four standard views: **CC-L, CC-R, MLO-L, MLO-R**.
-   DICOM header information from the original mammograms are provided as stacked `.json` files, with separate stacks for each of the four standard views, synchronized with the mammograms.
-   The example algorithm predicts breast cancer risk at multiple time points (years 1–5), based on four DICOM files per participant.
-   Helper code is included to convert stacked `.mha` files to individual DICOM files and to populate required DICOM header fields, after reading from the `.json` files.

### 🐳 `Dockerfile`

Defines the Docker container used for deployment on Grand Challenge.

-   All required system and Python dependencies are installed during the image build — no separate local Python or conda environment is needed.
-   For help setting up Docker with GPU support, see the [Grand Challenge documentation](https://grand-challenge.org/documentation/setting-up-wsl-with-gpu-support-for-windows-11/) or the [Docker documentation](https://docs.docker.com/engine/install/ubuntu/).

------------------------------------------------------------------------

## ⚙️ Configuration

Before building or running the container, open `inference.py` and verify the following paths:

| Variable | Description | Default |
|------------------------|------------------------|------------------------|
| `INPUT_PATH` | Path to the input `.mha` stacks (`IMAGES_PATH` = `INPUT_PATH/"images"`). For the .json files containing header information, their pat equals their filename | `/input` (keep for Grand Challenge) |
| `RESOURCE_PATH` | Path to resources such as model weights | `/opt/algorithm/resources` (see Dockerfile) |
| `OUTPUT_PATH` | Path where predictions will be written | `/output` (keep for Grand Challenge) |

------------------------------------------------------------------------

## 🚀 Running Locally

To test your algorithm locally before submission, run:

``` bash
./do_test_run.sh
```

This script will: - Launch the Docker container - Mount the required input and output directories - Run `inference.py` inside the container

Input is read from `./test/input` and output is written to `./test/output`.

------------------------------------------------------------------------

## 📤 Building for Submission

To build and export the Docker container for upload to Grand Challenge, run:

``` bash
./do_save.sh
```

This produces a `.tar.gz` file that can be uploaded directly to Grand Challenge. For more information on testing and submitting your container, see the [Grand Challenge documentation](https://grand-challenge.org/documentation/add-the-algorithm/).

------------------------------------------------------------------------
