import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd

# Closest color name (optional)
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
        dist = sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(rgb_tuple, rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

st.title("🔬 High-Fidelity Pixel Color Analysis (Medical Images)")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    top_n = st.slider("Top N Dominant Colors", 2, 50, 10)

    with st.spinner("Analyzing every pixel exactly..."):
        img_np = np.array(image)
        pixels = img_np.reshape(-1, 3)
        pixel_list = [tuple(pixel) for pixel in pixels]

        counter = Counter(pixel_list)
        total_pixels = sum(counter.values())
        most_common = counter.most_common(top_n)
        top_colors = set([color for color, _ in most_common])

        data_rows = []
        for i, (color, count) in enumerate(most_common):
            pct = (count / total_pixels) * 100
            name = closest_color_name(color)
            mean_intensity = int(np.mean(color))
            hex_code = '#%02x%02x%02x' % color

            data_rows.append({
                "Rank": i + 1,
                "% of Image": round(pct, 3),
                "RGB": color,
                "Hex": hex_code,
                "Closest Name": name,
                "Mean Intensity": mean_intensity
            })

        df = pd.DataFrame(data_rows)

        # Reconstruct full image using the closest match from top-N
        color_map = {color: color for color in top_colors}
        default_color = (0, 0, 0)
        recolored_img = np.zeros_like(img_np)

        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                pixel = tuple(img_np[i, j])
                if pixel in color_map:
                    recolored_img[i, j] = color_map[pixel]
                else:
                    # Find the closest color in top_colors
                    closest = min(top_colors, key=lambda c: sum((p - q) ** 2 for p, q in zip(c, pixel)))
                    recolored_img[i, j] = closest

        recolored_pil = Image.fromarray(recolored_img)

    st.subheader("📊 Exact Pixel Color Summary")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "pixel_colors.csv", "text/csv")

    st.subheader("🖼️ Reconstructed Image After Analysis")
    st.image(recolored_pil, caption="AI-Reconstructed Image", use_column_width=True)
