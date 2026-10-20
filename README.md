# clip_cam

`clip_cam` is a Python package for visualizing **the image-prompt feature matching** in ViT-based CLIP models, highlighting the alignment between image features and textual prompts. It allows you to visualize how CLIP interprets the relationship between an image and a text description, providing insights into its attention patterns.

## 🚀 Features

- Generate Grad-CAM-style heatmaps for image-text matching, computed as dense image-text similarity (no gradients needed).
- Support for Vision Transformer (ViT) architectures: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `ViT-L/14@336px`.
- Easy integration with existing CLIP implementations.
- Custom checkpoint support for fine-tuned models.

## 📦 Installation

You can install `clip_cam` via pip:

```bash
pip install clip_cam
```

Or install the latest version directly from GitHub:

```bash
pip install git+https://github.com/adityagandhamal/clip_cam
```

## 🔥 Usage

Run the following command to generate a visualization:

```bash
clip-cam --model_name "ViT-B/16" --image_path "path/to/image.jpg" --text "your text prompt"
```

If you cloned the repo instead, you can run `python clip_cam.py` with the same arguments.

### Arguments:
- `--image_path`: Path to the input image.
- `--text`: Text input/prompt.
- `--model_name`: CLIP model name (default: `ViT-B/16`).
- `--checkpoint`: (Optional) Path to a fine-tuned CLIP model checkpoint.
- `--resolution`: Input image resolution (default: `384`). Higher values give sharper heatmaps but run slower.

### Example:
```bash
clip-cam --model_name "ViT-B/16" --image_path "assets/cat.jpg" --text "a cute kitten"
```

## 🛠️ How It Works

1. **Model Loading**: Uses the specified CLIP model with optional fine-tuned checkpoint.
2. **Feature Extraction**:
   - Extracts dense visual features from the image.
   - Encodes the text prompt and normalizes the embeddings.
3. **Matching & Visualization**:
   - Computes image-text matching scores.
   - Resizes the matching map using bilinear interpolation.
   - Visualizes the results generating a heatmap of image-text matching.


## 🔥 Example Visualization

![Sample Output](https://raw.githubusercontent.com/adityagandhamal/clip_cam/main/assets/clip_cam.png)  
_Example visualization showing the similarity heatmap over the image for the provided text prompt._

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

