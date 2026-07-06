Multimodal Wildlife Localization
Overview
Locate the animal in each paired RGB and thermal aerial wildlife crop. Each test row corresponds to one real AWIR crop containing a single cow, deer, or horse after deterministic crop/resize/flip preprocessing. Predict a normalized bounding box around the visible animal using the provided public training boxes and the paired image arrays.

The challenge is designed for CPU-feasible computer vision and multimodal feature extraction. The RGB crop contains texture and shape cues, while the thermal crop often helps separate the animal from grass, soil, and shadows.

Dataset
File descriptions:

dataset/public/train.csv: Training IDs, array indices, species labels, and normalized bounding boxes.
dataset/public/test.csv: Test IDs and array indices. Bounding boxes are hidden.
dataset/public/train/images.npz: Training image arrays with keys rgb and thermal; arrays are aligned to array_index in train.csv.
dataset/public/test/images.npz: Test image arrays with keys rgb and thermal; arrays are aligned to array_index in test.csv.
dataset/public/sample_submission.csv: A valid random submission in the required format.
Column descriptions:

id: Hashed example identifier.
array_index: Row position in the matching NPZ arrays.
class_label: Animal class for public training examples only.
x_min: Left edge of the bounding box, normalized from 0 to 1.
y_min: Top edge of the bounding box, normalized from 0 to 1.
x_max: Right edge of the bounding box, normalized from 0 to 1.
y_max: Bottom edge of the bounding box, normalized from 0 to 1.
Evaluation
Submissions are scored with composite bounding-box localization loss. Lower is better.


intersection = area(predicted_box & true_box)

union = area(predicted_box | true_box)

iou = intersection / union

coord_mae = mean(abs(predicted_coords - true_coords))

row_loss = 100  *(0.65*  (1 - iou) + 0.35 * coord_mae)

score = mean(row_loss)

Invalid submissions are rejected. Coordinates must be finite normalized values between 0 and 1, and every row must satisfy x_min < x_max and y_min < y_max.

Submission
Submission columns:

id: Test example identifier.
x_min: Predicted normalized left edge.
y_min: Predicted normalized top edge.
x_max: Predicted normalized right edge.
y_max: Predicted normalized bottom edge.
Example:


id,x_min,y_min,x_max,y_max

AWIR_f4295657b38f,0.1200,0.1800,0.7300,0.6900

Requirements:

Submit exactly one row for every test ID.
Use the same IDs as dataset/public/test.csv.
Do not use external AWIR source metadata, original image filenames, or private labels.
Write predictions to ./working/submission.csv.