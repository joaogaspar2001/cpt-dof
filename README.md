# Computational Depth-of-Field

This project focuses on the implementation of Computational Depth-of-Field using various techniques and tools. The primary objective is to simulate the depth-of-field effect, which is a characteristic of optical systems, using computational methods.

Depth-of-Field (DoF) refers to the range in a scene where the objects appear acceptably sharp in an image. In photography, it's used to draw attention to specific subjects in the frame by blurring the foreground and/or background. This project aims to achieve this effect computationally, allowing for post-processing adjustments to the perceived depth in an image.

## Files

- `bilateral_filter.py`: A module containing functionality related to the bilateral filtering process. This filtering method is used to reduce noise while preserving edges.
- `depth_map_CNN.py`: A module related to estimating the depth map using the [MiDaS 3.1 `dpt_swin2_tiny_256` model](https://github.com/isl-org/MiDaS) CNN.
- `depth_map.py`: A module for generating and working with depth maps by solving the constrained Poisson equation.
- `helpers.py`: A utility module that contains helper functions, which are commonly used across different parts of the application.
- `ken_burns.py`: A module related to the Ken Burns effect.
- `main.py`: The main entry point of the application.
- `ui.py`: A module related to the user interface of the application.

## Setup

This code is written in `Python 3.10.8` and requires the packages listed in `requirements-cpu.yaml` (for CPU-based execution) or `requirements-gpu.yaml` (for GPU usage).

### CPU-based execution

```bash
# Create a virtual environment using conda
conda env create --file environment-cpu.yaml

# Activate the virtual environment
conda activate cpt-dof-cpu
```

### GPU usage

```bash
# Create a virtual environment using conda
conda env create --file environment-gpu.yaml

# Activate the virtual environment
conda activate cpt-dof-gpu
```

## Usage

```
python src/main.py
```

Use the associated command-line arguments to specify parameters:

- Initial image file path to load: `-F/--file`
- Enable model optimization for CUDA when using the CNN depth map estimation (only works if CUDA exists): `-o/--optimize`
- Warping factor (in %) for the Ken Burns effect: `-w/--warping-factor`
- Near plane distance (millimeters) for the Ken Burns effect: `-n/--near-plane`
- Far plane distance (millimeters) for the Ken Burns effect: `-f/--far-plane`
- Duration of the video (seconds) for the Ken Burns effect: `-d/--duration-video`
- Enable anaglyph mode when using the Ken Burns effect: `-a/--anaglyph`

Note that all the possible command line arguments can be consulted using `python src/main.py -h` or `python src/main.py --help`.

## Examples

In this section can be found examples outputed by this application. For more examples and detailed view of this project there is also a [report](https://drive.google.com/file/d/1iUSPij2tqwtIhYTOwSpba4EHQAdfxMsF/view?usp=drive_link) and a [video](https://drive.google.com/file/d/1S1vXWg65nFXsTl3Nj-wsz6OxNPXEOyec/view?usp=drive_link).

### Example 1 - Arch

<div align="center">
    <img src="/examples/arch.gif" width="400">
</div>

**Legend:** 3D Ken-Burns effect applied on arch image.

### Example 2 - Firefighter

<div align="center">
    <img src="/examples/firefighter.gif" width="400">
</div>

**Legend:** 3D Ken-Burns effect applied on firefigther image.

### Example 3 - Miami Rescue

<div align="center">
    <img src="/examples/miamirescue.gif" width="400">
</div>

**Legend:** 3D Ken-Burns effect applied on Miami rescue image.

### Example 4.1 - Art

<div align="center">
    <img src="/examples/art.gif" width="400">
</div>

**Legend:** 3D Ken-Burns effect applied on art image.

### Example 4.2 - Art

<div align="center">
    <img src="/examples/art2.gif" width="400">
</div>

**Legend:** Stereo Ken-Burns effect applied on art image produced by using `python src/main.py -a`. Put your 3D glasses on!