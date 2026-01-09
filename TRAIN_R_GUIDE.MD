# train_r.py Guide



## High Level Explanation of the Code 
It trains the network to estimate relative 3D rotation between two images, without translation, using the direction-distribution formulation introduced in the paper.



## Mapping Code Outputs to Paper Variables

| Paper concept                      | Implemented here             |
| ---------------------------------- | ---------------------------- |
| DirectionNet-R                     | `model.DirectionNet`         |
| Directional outputs                | `n_out = 3 (9D) or 2 (6D)`   |
| Directional distribution           | vMF on discretized sphere    |
| Direction loss + distribution loss | Combined training objective  |
| Rotation reconstruction            | Gram–Schmidt or SVD          |
| Training loop                      | TF1 MonitoredTrainingSession |
