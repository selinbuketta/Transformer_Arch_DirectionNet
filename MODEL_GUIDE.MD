# model.py Guide

At a high level, model.py is the full neural architecture of DirectionNet, i.e., the model that takes two panoramic images and predicts one or more spherical probability distributions over directions (e.g., camera-to-camera relative direction or correspondence likelihood on the sphere).

This file corresponds directly to the “Overall Architecture” + “Directional Learning Component” sections of the paper.

## High Level Explanation of the Code 

(img1, img2)
   ↓
Joint Encoder (Siamese / Transformer)
   ↓
Global embedding
   ↓
Spherical upsampling + residual decoding
   ↓
Dense directional probability maps



## Mapping Code Outputs to Paper Variables

1. Core paper variable: spherical directional distribution

| Paper concept              | Code tensor                |
| -------------------------- | -------------------------- |
| Discretized sphere         | 64×64 equirectangular grid |
| Directional PDF            | `down_channel(y)`          |
| Multiple hypotheses        | channel dimension `N`      |
| Conditioning on image pair | encoder(img1, img2)        |

2. Variable: latent representation of image pair

| Paper symbol                   | Code              |
| ------------------------------ | ----------------- |
| Joint embedding ( \mathbf{z} ) | `y` after encoder |
| Learned parameters ( \theta )  | encoder weights   |
| View-pair conditioning         | Siamese + concat  |

3. Variable: spherical discretization grid

| Paper concept         | Code mechanism            |
| --------------------- | ------------------------- |
| Sphere continuity     | `equirectangular_padding` |
| Longitude wrap        | horizontal padding        |
| Resolution refinement | repeated upsampling       |
| Spherical grid        | tensor height × width     |

4. Variable: number of predicted distributions

| Paper               | Code               |
| ------------------- | ------------------ |
| ( N ) distributions | `n_out`            |
| ( P_k(\mathbf{d}) ) | output channel `k` |
| Joint prediction    | shared decoder     |

5. Variable: directional likelihood before normalization

| Paper                | Code             |
| -------------------- | ---------------- |
| Unnormalized logits  | raw CNN output   |
| Probability density  | enforced by loss |
| Direction likelihood | pixel value      |

| Paper variable    | Meaning         | Code tensor          |
| ----------------- | --------------- | -------------------- |
| ( I_1, I_2 )      | Input images    | `img1`, `img2`       |
| ( f_\theta )      | Encoder         | `self.encoder`       |
| ( \mathbf{z} )    | Joint embedding | encoder output       |
| ( \mathbb{S}^2 )  | Sphere          | equirectangular grid |
| ( P(\mathbf{d}) ) | Directional PDF | final output         |
| ( N )             | # hypotheses    | `n_out`              |
