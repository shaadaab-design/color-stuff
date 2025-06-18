import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import pandas as pd
from skimage import measure, color
from skimage.filters import threshold_otsu
from skimage.color import label2rgb

# Hardcoded CSS color names
def closest_color_name(rgb_tuple):
    css3_names_to_rgb = {
        'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
        'orange': (255, 165, 0), 'pink': (255, 192, 203), 'purple': (128, 0, 128),
        'brown': (165, 42, 42), 'lime': (0, 255, 0), 'navy': (0, 0, 128),
        'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'teal': (0, 128, 128),
        'silver': (192, 192, 192),
    }

    min_dist = float("inf")
    closest_name = None
    for name, rgb in css3_names_to_rgb.items():
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

# Particle analyzer with automatic settings
def particle_analysis_auto(image, filename):
    gray_img = image.convert("L")
    img_np_gray = np.array(gray_img)
    img_np_color = np.array(image.convert("RGB"))

    threshold = threshold_otsu(img_np_gray)
    binary = img_np_gray > threshold

    labels_img = measure.label(binary)
    props = measure.regionprops(labels_img, intensity_image=img_np_gray)

    particle_colors = []
    particle_areas = []

    for prop in props:
        if prop.area < 20:  # auto min area
            continue
        coords = prop.coords
        mean_color = img_np_color[coords[:, 0], coords[:, 1]].mean(axis=0)
        particle_colors.append(mean_color)
        particle_areas.append(prop.area)

    if len(particle_colors) == 0:
        return [], 0, None

    particle_colors = np.array(particle_colors)
    lab_colors = color.rgb2lab(particle_colors.reshape(-1, 1, 3).astype(np.uint8) / 255.0).reshape(-1, 3)

    kmeans = KMeans(n_clusters=min(6, len(lab_colors)), n_init=10, random_state=42)
    labels = kmeans.fit_predict(lab_colors)

    groups = defaultdict(lambda: {"count": 0, "total_area": 0, "mean_color_sum": np.array([0, 0, 0], dtype=float)})

    for idx, label in enumerate(labels):
        groups[label]["count"] += 1
        groups[label]["total_area"] += particle_areas[idx]
        groups[label]["mean_color_sum"] += particle_colors[idx] * particle_areas[idx]

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

    overlay = label2rgb(labels_img, image=np.array(image), bg_label=0)

    return data_rows, len(props), overlay


# ---------- STREAMLIT INTERFACE ----------
st.title("🧪 Auto Particle Analyzer")
st.write("Just upload an image. No sliders. No manual work. Fully automatic and accurate.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing particles..."):
        rows, total_particles, overlay = particle_analysis_auto(image, uploaded_file.name)

    st.subheader("🧾 Total Particles Detected")
    st.write(f"**{total_particles}** particles found.")

    if overlay is not None:
        st.subheader("🖼️ Particle Detection Preview")
        st.image((overlay * 255).astype(np.uint8), caption="Overlay of Detected Particles", use_column_width=True)

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("📊 Particle Group Summary")
        st.dataframe(df)
    else:
        st.warning("No particles found. Try uploading a clearer image.")


