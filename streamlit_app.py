import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict
import pandas as pd

from skimage import measure, color
from skimage.filters import threshold_otsu
from skimage.color import label2rgb

# Simple color names dictionary for matching cluster colors
CSS3_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'purple': (128, 0, 128),
    'brown': (165, 42, 42),
    'lime': (0, 255, 0),
    'navy': (0, 0, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'teal': (0, 128, 128),
    'silver': (192, 192, 192),
}

def closest_css3_color(rgb):
    """Find the closest CSS3 color name by Euclidean distance"""
    min_dist = float('inf')
    closest_name = None
    for name, c_rgb in CSS3_COLORS.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(c_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def analyze_particles(image, min_area, threshold, n_clusters=6):
    # Convert to grayscale and numpy array
    gray_img = image.convert('L')
    img_gray_np = np.array(gray_img)
    img_color_np = np.array(image.convert('RGB'))

    # Auto threshold with Otsu if threshold=0 or None
    if threshold is None or threshold == 0:
        threshold = threshold_otsu(img_gray_np)

    # Create binary mask: True = particle, False = background
    binary_mask = img_gray_np > threshold

    # Label connected components (particles)
    labels = measure.label(binary_mask)

    # Get properties of labeled regions
    regions = measure.regionprops(labels, intensity_image=img_gray_np)

    particle_colors = []
    particle_areas = []

    for region in regions:
        if region.area < min_area:
            continue  # skip small noise
        coords = region.coords
        mean_color = img_color_np[coords[:,0], coords[:,1]].mean(axis=0)
        particle_colors.append(mean_color)
        particle_areas.append(region.area)

    if len(particle_colors) == 0:
        return [], 0, None

    particle_colors = np.array(particle_colors)

    # Cluster particle colors in RGB space
    k = min(n_clusters, len(particle_colors))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(particle_colors)

    groups = defaultdict(lambda: {"count": 0, "total_area": 0, "color_sum": np.array([0.0,0.0,0.0])})

    for idx, cluster_label in enumerate(labels_k):
        groups[cluster_label]["count"] += 1
        groups[cluster_label]["total_area"] += particle_areas[idx]
        groups[cluster_label]["color_sum"] += particle_colors[idx] * particle_areas[idx]

    total_image_area = img_gray_np.shape[0] * img_gray_np.shape[1]
    data_rows = []

    for label, data in groups.items():
        count = data["count"]
        total_area = data["total_area"]
        avg_size = total_area / count if count else 0
        percent_area = (total_area / total_image_area) * 100
        mean_color = (data["color_sum"] / total_area).astype(int)
        color_name = closest_css3_color(mean_color)

        data_rows.append({
            "Color Group": f"{color_name}",
            "Count": count,
            "Total Area": round(total_area, 2),
            "Average Size": round(avg_size, 2),
            "Percentage Area": round(percent_area, 3),
            "Mean RGB": tuple(mean_color)
        })

    # Overlay for visualization: label2rgb overlays colored regions on original image
    overlay_img = label2rgb(labels, image=img_color_np, bg_label=0)

    return data_rows, len(regions), overlay_img

# ---- Streamlit UI ----

st.title("🧪 Particle Analyzer with Color Clustering")
st.write("Upload an image to detect particles, cluster them by color, and analyze size.")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("Settings")
    min_area = st.slider("Minimum particle area (pixels)", 1, 500, 20)
    threshold = st.slider("Threshold (0 = auto Otsu)", 0, 255, 0)
    n_clusters = st.slider("Number of color clusters", 1, 10, 6)

    actual_threshold = None if threshold == 0 else threshold

    with st.spinner("Analyzing..."):
        rows, total_particles, overlay = analyze_particles(img, min_area, actual_threshold, n_clusters)

    st.subheader(f"Particles detected (before filtering): {total_particles}")

    if overlay is not None:
        st.image((overlay * 255).astype(np.uint8), caption="Particle detection overlay", use_column_width=True)

    if rows:
        st.subheader("Particle Color Group Summary")
        df = pd.DataFrame(rows)
        st.dataframe(df)
    else:
        st.warning("No particles detected with current settings. Try lowering minimum area or threshold.")

