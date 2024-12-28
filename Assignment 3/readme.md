# SFM and Gaussian Splatting

## Overview

This involves two main tasks:

- Using COLMAP for Structure from Motion (SFM) to infer camera poses and reconstruct 3D positions of sparse points from images.
- Using Gaussian Splatting (GS) to train a model with the outputs of COLMAP and render images from novel viewpoints.

## Task 1: Structure from Motion with COLMAP

Prerequisites

- Install COLMAP from [GitHub](https://github.com/colmap/colmap). Use pre-built binaries or build from source for your OS (Windows, Linux, or MacOS).
- Download the a dataset, which contains 20 images.

Steps

- Dataset Preparation:

  - Divide the dataset into two sets:

    - Set 1: First 10 images.
    - Set 2: All 20 images.

- SFM Configuration:

     - Use the SIMPLE PINHOLE camera model.
     - Try both exhaustive and sequential matching options.

- Running COLMAP:

  - Process the datasets using COLMAP.
  - Generate outputs, including camera poses and reconstructed 3D points.

- Visualization:

  - Use COLMAP’s visualization tools to inspect:
    - Detected feature points.
    - Matched points.
    - Camera locations.
  - Evaluate the quality of results qualitatively.

- Quantitative Evaluation:

  - Retrieve the reprojection error by selecting "Extras > Show model statistics" in the GUI.
  - Compare the results across:
    - Matching modes (exhaustive vs. sequential).
    - Dataset sizes (10 images vs. 20 images).

## Task 2: Gaussian Splatting

Prerequisites

- Use the COLMAP outputs (camera poses and sparse 3D points).
- Install Gaussian Splatting code from [this repository](https://github.com/graphdeco-inria/gaussian-splatting).
- For easy setup on Colab, refer to [this Colab implementation](https://github.com/camenduru/gaussian-splatting-colab).

Steps

- Training the GS Model:

  - Experiment 1:

    - Use the first 10 images as the training set, excluding "image #5."
    - Train the GS model and render "image #5" using its camera pose.
    - Compare the rendered image with the actual image and compute quantitative metrics.
    - Note: "image #5" is used in COLMAP for sparse 3D points, so results may not be entirely independent.

  - Experiment 2:

    - Use all 20 images for training, excluding "images #5, #10, and #15."
    - Train the GS model and render the excluded images using their respective camera poses.
    - Observe changes in results with a larger training set.

- Effect of Iterations:

  - Repeat both experiments above with different numbers of training iterations.
  - Analyze how the quality of rendering changes with iteration count.

- Visual and Quantitative Evaluation:

  - Render scenes from arbitrary viewpoints.
  - Separate images into training and test sets for evaluation.
  - Use metrics provided by the GS software for quantitative analysis.

## Expected Outputs

- COLMAP Outputs:

  - Camera poses and reconstructed 3D points for both datasets (10 images and 20 images).
  - Visualization of results (feature points, matched points, camera locations).
  - Reprojection error comparison.

- Gaussian Splatting Outputs:

  - Rendered images for the test set.
  - Quantitative metrics comparing rendered and actual images.
  - Analysis of changes with dataset size and number of iterations.

- Observations and Insights:

  - Comparison of results between exhaustive and sequential matching.
  - Impact of training set size on GS results.
  - Dependence of rendering quality on the number of GS training iterations.

## Tools and Resources

- [COLMAP GitHub Repository](https://github.com/colmap/colmap)
- [Gaussian Splatting GitHub Repository](https://github.com/graphdeco-inria/gaussian-splatting)
- [Gaussian Splatting Colab Implementation](https://github.com/camenduru/gaussian-splatting-colab)
