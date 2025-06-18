import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import pandas as pd

from skimage import measure
from skimage.filters import threshold_otsu

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def particle_analysis_grouped(image, filename, n_color_groups=3):
    gray_img = image.convert("L")
    img_np_gray = np.array(gray_img)
    img_np_color = np.array(image)

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

    kmeans = KMeans(n_clusters=n_color_groups)
    labels = kmeans.fit_predict(particle_colors)
    cluster_centers = kmeans.cluster_centers_.astype(int)

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
        hex_color = rgb_to_hex(mean_color)
        # mean intensity approximated as average of grayscale intensities weighted by area
        mean_intensity = int(np.mean(mean_color))  # or use another method if you want

        # Compose a label like: filename (color)
        color_name = f"RGB({mean_color[0]}, {mean_color[1]}, {mean_color[2]})"
        label = f"{filename} ({color_name})"

        data_rows.append({
            "Label": label,
            "Count": count,
            "Total Area": total_area,
            "Average Size": avg_size,
            "Percentage Area": percent_area,
            "Mean Intensity": mean_intensity,
            "Hex Color": hex_color
        })

    return data_rows

st.title("🎨 Particle Group Analysis Table")
st.write("Upload your image and see grouped particle stats in a table.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    filename = uploaded_file.name

    with st.spinner("Analyzing grouped particles..."):
        rows = particle_analysis_grouped(image, filename, n_color_groups=3)

    if rows is None or len(rows) == 0:
        st.write("No particles detected.")
    else:
        df = pd.DataFrame(rows)
        # Show color swatch instead of Hex Color string (optional)
        def color_cell(val):
            return f'background-color: {val}'

        st.subheader("Particle Groups Summary Table")
        # Show table with colors highlighted in 'Hex Color' column
        st.dataframe(df.style.applymap(color_cell, subset=["Hex Color"]))
