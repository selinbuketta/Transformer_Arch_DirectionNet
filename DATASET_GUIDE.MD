# dataset.py Guide

This file is the paper's data engine: it synthesizes realistic wide-baseline image pairs from panoramas and computes exact relative pose labels, enabling large-scale supervised learning of camera geometry without real stereo data.

## High Level Explanation of the Code 

dataset.py is the mechanism that turns panoramas into supervised training data. 

1. It starts from a panorama. 
2. Samples two virtual cameras.
3. Projects perspective images: Converts the panoroma into two standard pinhole-camera images
4. Computes perfect ground truth: Relative rotation (R2->1)

## Mapping Code Outputs to Paper Variables

| Code output          | Meaning in paper                                          |
| -------------------- | --------------------------------------------------------- |
| `src_img`, `trt_img` | Input image pair to the network                           |
| `r_c2_to_c1`         | Ground-truth relative rotation                            |
| `t_c1`               | Ground-truth relative translation direction               |
| `fov`                | Camera intrinsics parameter                               |
| `mask1`, `mask2`     | Visibility overlap (used for filtering or loss weighting) |

