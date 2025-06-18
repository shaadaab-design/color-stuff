import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import pandas as pd

from skimage import measure, color
from skimage.filters import threshold_otsu
import webcolors

def closest_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        min_dist = float("inf")
        closest_name = None
        for hex_val in webcolors.CSS3_NAMES_TO_HEX.values():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
            dist = ((r_c - rgb_tuple[0]) ** 2 + (g_c - rgb_tuple[1]) ** 2 + (b_c - rgb_tuple[2]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_name = webcolors.hex_to_name(hex_val)
        return closest_name

def particle_analysis_grouped(image, filename, n_color_groups=5):
    gray_img = image.convert("L")
    img_np_gray = np.array(gray_img)
    img_np_color = np.array(image.convert("RGB"))  # Ensure RGB

    thresh = threshold_otsu(img_np_gray)
    binary = img_np_gray > thresh

    labels_img = measure.label(binary)
    props = measure.regionprops(labels_img, intensity_image=img_np_gray)

    if len(props) == 0:
        return None

    particle_colors = []
    particle_areas = []

    for prop in props:
        coords = prop.coords
        mean_color = img_np_color[coords[:,0], coords[:,1]].mean(axis=0)
        particle_colors.append(mean_color)
        particle_areas.append(prop.area)

    particle_colors = np.array(particle_colors)

    # Convert to LAB color space for perceptual accuracy
    lab_colors = color.rgb2lab(particle_colors.reshape(-1, 1, 3).astype(np.uint8) / 255.0).reshape(-1, 3)

    kmeans = KMeans(n_clusters=min(n_color_groups, len(lab_colors)), n_init=10, random_state=42)
    labels = kmeans.fit_predict(lab_colors)

    groups = defaultdict(lambda: {"count":0, "total_area":0, "mean_color_sum":np.array([0,0,0], dtype=float)})

    for idx, cluster_label in enumerate(labels):
        groups[cluster_label]["count"] += 1
        groups[cluster_label]["total_area"] += particle_areas[idx]
        groups[cluster_label]["mean_color_sum"] += particle_colors[idx] * particle_areas[idx]

    total_image_area = img_np_gray.shape[0] * img_np_gray.shape[1]

    data_rows = []

    for group_label, data in groups.items():
        count = data["count"]
        total_area = data["total_area"]
        avg_size = total_area / count if count > 0 else 0
        percent_area = (total_area / total_image_area) * 100
        mean_color = (data["mean_color_sum"] / total_area).astype(int)
        color_name = closest_color_name(mean_color)
        mean_intensity = int(np.mean(mean_color))

        label = f"{filename} ({color_name})"

        data_rows.append({
            "Label": label,
            "Color": color_name,
            "Count": count,
            "Total Area": round(total_area, 3),
            "Average Size": round(avg_size, 3),
            "Percentage Area": round(percent_area, 3),
            "Mean Intensity": mean_intensity
        })

    return data_rows

st.title("🧪 Particle Group Analysis (Named Colors)")
st.write("Upload an image (e.g. TIFF) and get detailed particle group stats by color.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    filename = uploaded_file.name

    with st.spinner("Analyzing grouped particles..."):
        rows = particle_analysis_grouped(image, filename, n_color_groups=6)

    if rows is None or len(rows) == 0:
        st.error("No particles detected.")
    else:
        df = pd.DataFrame(rows, columns=["Label", "Color", "Count", "Total Area", "Average Size", "Percentage Area", "Mean Intensity"])
        st.subheader("📋 Particle Summary Table")
        st.dataframe(df)
