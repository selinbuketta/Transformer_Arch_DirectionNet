# dataset_loader.py Guide



## High Level Explanation of the Code 

In the paper, DirectionNet is trained to infer relative camera motion geometry from a stereo image pair, specifically:

Rotation 
R∈SO(3)
R∈SO(3)

Translation direction / epipole

Field of View (FoV)

This data_loader is the bridge between the paper’s mathematical formulation and the actual training process. Its role is to:

1. Load paired perspective images (source & target views)

2. Load ground-truth geometric supervision:

-Rotation matrix

-Translation direction (epipole)

-Field of View

3. Optionally load predicted rotation from a previous model stage

4. Apply controlled photometric augmentation

5. Package everything into a batched TensorFlow dataset usable by the training loop

## Mapping Code Outputs to Paper Variables

| File                              | Meaning in the paper                  |
| --------------------------------- | ------------------------------------- |
| `*.src.perspective.png`           | Source image (I_s)                    |
| `*.trt.perspective.png`           | Target image (I_t)                    |
| `rotation_gt.pickle`              | Ground-truth rotation matrix (R_{gt}) |
| `epipoles_gt.pickle`              | Translation direction / epipole       |
| `fov.pickle`                      | Camera field of view                  |
| `rotation_pred.pickle` (optional) | Rotation estimated by DirectionNet-R  |
