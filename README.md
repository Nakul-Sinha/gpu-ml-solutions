# GPU ML Solutions

Three machine learning problems I solved on a single GPU, each with its own
folder, notes and solution code.

| Folder | What the problem is |
|---|---|
| `Multimodal Wildlife Localization/` | Predict a normalized bounding box around the animal in paired RGB and thermal aerial crops, each holding one cow, deer or horse |
| `Shadow-Based GPS and Time Prediction/` | Recover latitude, longitude and UTC hour from a single outdoor photo, using shadow length, direction and sharpness as the only reliable cues |
| `chess move prediction/` | Map a SAN move prefix to a calibrated win, draw and loss distribution for the cohort of games sharing that prefix |

The shadow one is the odd and interesting member of the set. It is an inverse sun
position problem: I am reading a 2D projection of 3D shadow geometry and trying
to invert it back to a place and a time on Earth.

Datasets are not committed. Each folder holds its own description and approach
notes.
