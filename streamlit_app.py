import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict
import pandas as pd
from skimage import measure, color
from skimage.color import label2rgb

def closest_color_name(rgb_tuple):
    css3_names_to_rgb = {
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

    min_dist = float("inf")
    closest_name = None
    for name, rgb in css3_names_to_rgb.items():
        dist = sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(rgb_tuple, rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def particle_analysis_color_mask(image, filename, n_color_groups=5, min_area=10, hue_range=(0, 360)):
    img_np = np.array(image.convert("RGB"))
    img_hsv = np.array(image.convert("HSV"))

    # Select pixels within hue range (in degrees 0-360)
    hue = img_hsv[:, :, 0] * 360 / 255  # scale hue to degrees
    mask = (hue >= hue_range[0]) & (hue <= hue_range[1])

    # Label connected regions in mask
    labels_img = measure.label(mask)
    props = measure.regionprops(labels_img)

    particle_colors = []
    particle_areas = []

    for prop in props:
        if prop.area < min_area:
            continue
        coords = prop.coords
        mean_color = img_np[coords[:, 0], coords[:, 1]].mean(axis=0)
        particle_colors.append(mean_color)
        particle_areas.append(prop.area)

    if len(particle_colors) == 0:
        return [], 0, None

    particle_colors = np.array(particle_colors)

    kmeans = KMeans(n_clusters=min(n_color_groups, len(particle_colors)), n_init=10, random_state=42)
    labels = kmeans.fit_predict(particle_colors)

    groups = defaultdict(lambda: {"count": 0, "total_area": 0, "mean_color_sum": np.array([0,0,0], dtype=float)})

    for idx, cluster_label in enumerate(labels):
        groups[cluster_label]["count"] +=1
        groups[cluster_label]["total_area"] += particle_areas[idx]
        groups[cluster_label]["mean_color_sum"] += particle_colors[idx] * particle_areas[idx]

    total_image_area = img_np.shape[0] * img_np.shape[1]
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

    overlay = label2rgb(labels_img, image=img_np, bg_label=0)
    return data_rows, len(props), overlay

# --- Streamlit UI ---
st.title("Particle Analyzer: Color-based Segmentation")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Settings")
    min_area = st.slider("Min particle area", 1, 500, 20)
    hue_min = st.slider("Hue min (degrees)", 0, 360, 0)
    hue_max = st.slider("Hue max (degrees)", 0, 360, 60)

    rows, total_particles, overlay = particle_analysis_color_mask(image, uploaded_file.name,
                                                                 min_area=min_area,
                                                                 hue_range=(hue_min, hue_max))

    st.write(f"Total particles detected: **{total_particles}**")

    if overlay is not None:
        st.image((overlay * 255).astype(np.uint8), caption="Particle overlay", use_container_width=True)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
    else:
        st.warning("No particles detected. Adjust hue range or min area.")



