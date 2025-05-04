**Writing README for scripts**

The user needs a README for three Python scripts. The README should cover the purpose of each script, how to install dependencies, how to run them, and possibly usage examples. I'll format the README in Markdown. For the three scripts, I'll include a brief description and maybe use citations to bring in content from the files where relevant. It's important to keep the documentation clear and straightforward to help users understand and run the scripts. Let's get that started!


Thought for 4 seconds


# Iris Processing Suite

This repository provides three standalone Python scripts for iris recognition research:

1. **Iris Segmentation** (`iris_segmentation.py`)
2. **Iris Localization (Daugman’s Operator)** (`iris_localization_daugman.py`)
3. **Iris Feature Extraction** (`iris_feature_extraction.py`)

---

## Requirements

* Python 3.6+
* [NumPy](https://pypi.org/project/numpy/)
* [OpenCV-Python](https://pypi.org/project/opencv-python/)
* [PyWavelets](https://pypi.org/project/PyWavelets/) (for wavelet features)

You can install the core dependencies with:

```bash
pip install numpy opencv-python PyWavelets
```

---

## Installation

Clone or download this repo, then ensure your `PATH` or virtual environment includes the above dependencies.

```bash
git clone https://github.com/yourusername/iris-processing-suite.git
cd iris-processing-suite
```

---

## Usage

### 1. Iris Segmentation

Segments the iris from an **eye crop** using Otsu’s thresholding + contour detection, and produces a binary mask plus an overlay on the original image .

```bash
python iris_segmentation.py \
  --image eye.jpg \
  --mask  iris_mask.png \
  --overlay iris_overlay.png
```

* **`--image`**: Path to input eye image (must be zoomed in on the eye).
* **`--mask`**: Output path for the binary iris mask.
* **`--overlay`**: Output path for the iris-only overlay.

---

### 2. Iris Localization (Daugman’s Operator)

Finds the **pupil** (inner) and **iris** (outer) boundaries via a brute-force integro-differential search on gradient magnitude .

```bash
python iris_localization_daugman.py \
  --image        eye.jpg \
  --inner-r-min  20  --inner-r-max  60 \
  --outer-r-min  60  --outer-r-max 120 \
  --center-range 30  --step        2  --res 360 \
  --output       overlay.png
```

* **`--image`**: Input eye crop.
* **`--inner-r-min/max`**: Radius search range for the **pupil** (px).
* **`--outer-r-min/max`**: Radius search range for the **iris** (px).
* **`--center-range`**: How many pixels around the image center to search.
* **`--step`**: Pixel step size for center offsets.
* **`--res`**: Number of angular samples per circle.
* **`--output`**: Path for the result overlay (pupil in green, iris in red).

---

### 3. Iris Feature Extraction

Normalizes a segmented iris into a rectangular “rubber-sheet,” then extracts:

* **Gabor-filter** responses (Daugman-style iris code)
* **Wavelet** coefficients via PyWavelets&#x20;

```bash
python iris_feature_extraction.py \
  --image        eye_crop.jpg \
  --pupil-x      120 --pupil-y    130 --pupil-r    40 \
  --iris-x       120 --iris-y     130 --iris-r     80 \
  --radial-res   64  --angular-res 512 \
  --freqs        8 16 32 \
  --thetas       0 0.7854 1.5708 2.3562 \
  --wavelet      db2 --wavelet-level 3 \
  --output       iris_feats.npz
```

* **`--pupil-x/y/r`**, **`--iris-x/y/r`**: Coordinates & radii of pupil and iris circles.
* **`--radial-res`**, **`--angular-res`**: Dimensions of the normalized image.
* **`--freqs`**, **`--thetas`**: Gabor wavelengths (px) and orientations (rad).
* **`--wavelet`**, **`--wavelet-level`**: Wavelet type & decomposition depth.
* **`--output`**: `.npz` file containing `gabor` and `wavelet` arrays.

---

## Inspecting `iris_feats.npz`

Load the feature vectors easily in Python:

```python
import numpy as np

data = np.load("iris_feats.npz")
print("Keys:", data.files)              # ['gabor','wavelet']
gabor   = data["gabor"]   # 1D Gabor feature vector
wavelet = data["wavelet"] # 1D Wavelet feature vector
```

You can then match, classify, or export these arrays as needed.

---

## License & Contributing

Feel free to modify or extend these scripts. Pull requests and issue reports are welcome!
