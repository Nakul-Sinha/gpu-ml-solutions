Overview
Given a single outdoor photograph containing visible shadows, predict the GPS coordinates (latitude, longitude) and UTC hour the image was captured. The only reliable geolocation cues are shadow geometry, length, direction, sharpness, combined with sky appearance and terrain texture.

This is an inverse sun-position problem: from a 2D projection of 3D shadow-casting geometry, recover where on Earth and when the image was taken. The task requires implicit understanding of solar geometry, atmospheric optics, and terrain cues, all learned from visual features through computer vision.

Evaluation
Submissions are scored using a composite of Haversine distance for geolocation and circular mean absolute error for time. The composite score ranges from ~0 to 1 (higher is better).

Haversine Distance (Geolocation Error)
Given predicted coordinates (lat₁, lon₁) and ground truth (lat₂, lon₂):

Δ
lat
=
lat
2
−
lat
1
Δ
lon
=
lon
2
−
lon
1
Δlat=lat 
2
​
 −lat 
1
​
 Δlon=lon 
2
​
 −lon 
1
​
 

a
=
sin
⁡
2
(
Δ
lat
2
)
+
cos
⁡
(
lat
1
)
⋅
cos
⁡
(
lat
2
)
⋅
sin
⁡
2
(
Δ
lon
2
)
a=sin 
2
 ( 
2
Δlat
​
 )+cos(lat 
1
​
 )⋅cos(lat 
2
​
 )⋅sin 
2
 ( 
2
Δlon
​
 )

d
=
6371
×
2
×
arcsin
⁡
(
min
⁡
(
a
,
1.0
)
)
(km)
d=6371×2×arcsin( 
min(a,1.0)
​
 )(km)

Circular Hour Error (Time Error)
e
hour
=
min
⁡
(
∣
hour
1
−
hour
2
∣
,
24
−
∣
hour
1
−
hour
2
∣
)
e 
hour
​
 =min(∣hour 
1
​
 −hour 
2
​
 ∣,24−∣hour 
1
​
 −hour 
2
​
 ∣)

Composite Score
geoscore
=
1
1
+
d
‾
/
500
timescore
=
1
1
+
e
hour
‾
/
3
geoscore= 
1+ 
d
 /500
1
​
 timescore= 
1+ 
e 
hour
​
 
​
 /3
1
​
 

score
=
0.5
×
geoscore
+
0.5
×
timescore
score=0.5×geoscore+0.5×timescore
​
 

Where 
d
‾
d
  is the mean Haversine distance across all test samples, and 
e
hour
‾
e 
hour
​
 
​
  is the mean circular hour error. The reference values (500 km, 3 h) normalize each component to a ~[0, 1] range before averaging.

Anti-cheat: Individual samples with raw geo error exceeding 10,000 km (more than a quarter of Earth's circumference) have their error multiplied by 1.5 before computing the mean. This penalizes degenerate strategies such as predicting a fixed midpoint for all samples.

Expected baselines:

Random guessing (uniform lat/lon/hour): ~0.18 (mean geo error ~10,000 km, mean time error ~6 h)
Sample submission (all zeros): ~0.23 (lat=0, lon=0, hour=12.0, better than random)
Physics-aware model (implicit sun geometry): ~0.70+
Perfect prediction: 1.0
Dataset
The dataset consists of 8,000 fully synthetic outdoor images (5,600 train, 2,400 test) at 512×512 resolution generated through a physically-based Blender rendering pipeline with Nishita atmospheric sky models. Images are procedurally generated, each scene is uniquely rendered with randomized terrain, shadow-casting objects (buildings, trees, rocks), and atmospheric conditions. Sun elevation is constrained to 8 to 55° to ensure visible shadows across all samples. The only variables preserved across images are the physically accurate sun position and the ground-truth GPS coordinate assigned to that sun position.

Data Generation
Images are produced by a procedural rendering pipeline. For each scene, a GPS coordinate and UTC hour are sampled, and the corresponding sun position (elevation and azimuth) is computed from first-principles solar geometry using the day of year. The scene is then drawn with the sun placed at the correct angular position and shadows cast in the correct direction. Deliberate noise is injected into sun placement, shadow angle, and atmospheric appearance, this prevents extracting sun position analytically from image pixels and forces models to learn solar geometry implicitly.

Key Design Decisions
Day of year (doy) is provided in both train.csv and test.csv. Without it, the inverse-sun-position problem is underdetermined. In any real-world scenario, the date a photo was taken is known.
Scene-level metadata is stripped: internal parameters such as sun elevation, sun azimuth, and scene type are not included in any public file. The model must infer them from the image alone.
Train and test sets are disjoint, no location or image appears in both splits.
Public Files
**train.csv**: Training labels (5,600 rows)

image_id (str), Filename without extension, e.g. scene_000001, maps to train/scene_000001.png
latitude (float), GPS latitude in decimal degrees, range [-90, 90]
longitude (float), GPS longitude in decimal degrees, range [-180, 180]
hour (float), UTC hour of capture, range [0, 24)
doy (int), Day of year the photo was taken, range [1, 365]
**test.csv**: Test file list (2,400 rows)

image_id (str), Filename without extension, maps to test/{image_id}.png
doy (int), Day of year the photo was taken, range [1, 365]
**sample_submission.csv**: Submission template (2,400 rows)

image_id (str), Filename from test/
latitude (float), Placeholder value 0.0, range [-90, 90]
longitude (float), Placeholder value 0.0, range [-180, 180]
hour (float), Placeholder value 12.0, range [0, 24)
All placeholder values are neutral, they carry no information about the test set answers.

Image Properties
Resolution: 512×512 pixels, RGB
Format: PNG
Scene types: urban, rural, desert, forest, coastal (randomly assigned, not correlated with image_id or any public feature)
Sun elevation range: 8 to 55° (constrained to ensure visible shadows; sun never directly overhead)
Content: outdoor scenes with physically accurate Nishita sky, procedural terrain, 3D objects (buildings, trees, rocks, poles, cacti), and ray-traced cast shadows. Shadow geometry is the primary predictive signal.
Submission
Submit a CSV file with the following columns:

image_id (str), Filename from test/
latitude (float), Predicted latitude in [-90, 90]
longitude (float), Predicted longitude in [-180, 180]
hour (float), Predicted UTC hour in [0, 24)
Example (first 3 rows):

image_id,latitude,longitude,hour  
scene_000255,48.8566,2.3522,14.5  
scene_000413,-33.8688,151.2093,8.2  
scene_000005,35.6762,139.6503,19.0  
...  

(2,400 rows total)

Requirements:

Must contain exactly 2,400 rows (one per test sample)
Include header row with exact column names: image_id, latitude, longitude, hour
Latitude: float in [-90, 90]
Longitude: float in [-180, 180]
Hour: float in [0, 24)
A sample_submission.csv with neutral placeholders (0.0, 0.0, 12.0) is provided in the public dataset
Rules & Constraints
Allowed
Base models: Any architecture up to 13B parameters. Pre-trained weights allowed (ImageNet, etc.) but no models pre-trained specifically on geolocation tasks.
Fine-tuning: LoRA, QLoRA, full fine-tuning, or training from scratch.
Training data: ONLY the provided training set. No external images, GPS databases, elevation maps, or astronomical data.
Ensembles: Model ensembles are allowed. No restriction on combining predictions from multiple independently trained models.
Libraries: Any open-source ML/vision library (PyTorch, TensorFlow, OpenCV for basic image I/O only).
NOT Allowed
Physics-based solvers: ABSOLUTELY NO direct computation of sun position equations, shadow geometry formulas, or any analytical trigonometry to solve for coordinates. The model must learn solar geometry IMPLICITLY from training data. This includes:
No sun elevation/azimuth extraction from image pixels
No shadow angle measurement with image processing
No analytical solution of the sun-position equations
No hard-coded astronomical formulas or ephemeris data
External knowledge bases: No Wikipedia, astronomical tables, sun position calculators, GIS databases, or elevation maps.
Closed-source / commercial APIs: No GPT-4, Claude, Gemini, or any LLM API (local or cloud).
Human-written rules: No regex, hard-coded rules, or manual feature engineering for sun/shadow detection.
Data leak exploitation: Must not exploit any known data generation patterns. If you discover a leak, report it, don't exploit it.
 