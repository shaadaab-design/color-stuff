import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import colorsys
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Convert RGB to HSV (0-360 hue scale)
def rgb_to_hsv_pixel(rgb):
    r, g, b = rgb
    return tuple(round(i * 255) for i in colorsys.rgb_to_hsv(r / 255, g / 255, b / 255))

def classify_color_hue(hsv):
    h, s, v = hsv
    if s < 25 or v < 25:
        return "black"
    elif v > 230 and s < 30:
        return "white"
    elif s < 30:
        return "gray"
    if 0 <= h <= 15 or h >= 230:
        return "red"
    elif 16 <= h <= 45:
        return "orange"
    elif 46 <= h <= 75:
        return "yellow"
    elif 76 <= h <= 150:
        return "green"
    elif 151 <= h <= 200:
        return "cyan"
    elif 201 <= h <= 250:
        return "blue"
    elif 251 <= h <= 300:
        return "purple"
    else:
        return "pink"

st.title("🔬 High-Sensitivity Color Classification for Medical Imaging")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing pixel color categories with medical-level precision..."):
        img_np = np.array(image)
        flat_pixels = img_np.reshape(-1, 3)
        hsv_pixels = [rgb_to_hsv_pixel(rgb) for rgb in flat_pixels]
        labels = [classify_color_hue(hsv) for hsv in hsv_pixels]

        counter = Counter(labels)
        total = sum(counter.values())

        # Prepare particle table
        data_rows = []
        for name, count in counter.items():
            mask = np.array(labels) == name
            selected_pixels = flat_pixels[mask]
            mean_intensity = int(np.mean(selected_pixels)) if len(selected_pixels) > 0 else 0
            data_rows.append({
                "Particle": f"CTL LLLOS 467.tif ({name})",
                "Count": count,
                "Total Area": float(count),
                "Average Size": 1.0,
                "%Area": round((count / total) * 100, 3),
                "Mean": mean_intensity
            })

        df = pd.DataFrame(data_rows).sort_values("%Area", ascending=False)

    st.subheader("📊 Particle Summary Table (Hue-Classified)")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "hue_classified_summary.csv", "text/csv")

    # Region visualization for double-checking
    st.subheader("🧪 Visual Region Map: Confirm Detected Color Areas")
    selected_hue = st.selectbox("Select hue to visualize as a region mask", df["Particle"].str.extract(r'\((.*?)\)')[0].unique())

    # Generate mask and region overlay for selected hue
    label_array = np.array(labels).reshape(img_np.shape[0], img_np.shape[1])
    binary_mask = (label_array == selected_hue).astype(np.uint8)
    labeled = label(binary_mask)
    props = regionprops(labeled)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    for region in props:
        y0, x0, y1, x1 = region.bbox
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rect)
    ax.set_title(f"Detected regions for hue: {selected_hue}")
    ax.axis("off")
    st.pyplot(fig)

