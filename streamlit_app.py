import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
from skimage.measure import label, regionprops
from skimage.color import rgb2lab, deltaE_cie76
import matplotlib.pyplot as plt
import cv2

# Enhanced reference colors with better green representation
HUE_REF_COLORS = {
    "red": [(255, 0, 0), (220, 20, 60), (178, 34, 34), (255, 69, 0)],
    "orange": [(255, 165, 0), (255, 140, 0), (255, 69, 0)],
    "yellow": [(255, 255, 0), (255, 215, 0), (255, 255, 224)],
    "light_green": [(144, 238, 144), (152, 251, 152), (173, 255, 47), (154, 205, 50)],
    "medium_green": [(0, 255, 0), (50, 205, 50), (124, 252, 0), (127, 255, 0)],
    "dark_green": [(0, 128, 0), (34, 139, 34), (0, 100, 0), (85, 107, 47)],
    "forest_green": [(46, 125, 50), (76, 175, 80), (56, 142, 60)],
    "cyan": [(0, 255, 255), (64, 224, 208), (0, 206, 209)],
    "blue": [(0, 0, 255), (30, 144, 255), (70, 130, 180)],
    "purple": [(128, 0, 128), (138, 43, 226), (147, 112, 219)],
    "pink": [(255, 192, 203), (255, 20, 147), (219, 112, 147)],
    "brown": [(165, 42, 42), (139, 69, 19), (160, 82, 45)],
    "black": [(0, 0, 0), (25, 25, 25), (64, 64, 64)],
    "gray": [(128, 128, 128), (169, 169, 169), (105, 105, 105)],
    "white": [(255, 255, 255), (248, 248, 255), (240, 248, 255)]
}

def convert_colors_to_lab(color_dict):
    """Convert RGB color dictionary to Lab color space"""
    lab_dict = {}
    for name, rgb_list in color_dict.items():
        lab_dict[name] = []
        for rgb in rgb_list:
            # Ensure RGB values are in correct format
            rgb_array = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            lab_color = rgb2lab(rgb_array)[0][0]
            lab_dict[name].append(lab_color)
    return lab_dict

def classify_pixel_color(lab_pixel, ref_lab_colors, threshold=30):
    """
    Classify a pixel using Delta E distance in Lab color space
    Returns the best matching color name and distance
    """
    best_label = None
    best_distance = float("inf")
    
    for color_name, lab_refs in ref_lab_colors.items():
        for ref_lab in lab_refs:
            # Calculate Delta E distance
            distance = deltaE_cie76(lab_pixel.reshape(1, -1), 
                                  np.array(ref_lab).reshape(1, -1))[0]
            
            if distance < threshold and distance < best_distance:
                best_label = color_name
                best_distance = distance
    
    return best_label, best_distance

def preprocess_image(image, enhance_contrast=True):
    """Preprocess image for better color detection"""
    img_array = np.array(image)
    
    if enhance_contrast:
        # Convert to LAB and enhance
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced_lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return img_array

def create_color_statistics(pixel_labels, flat_pixels, image_name="image"):
    """Create comprehensive statistics for classified colors"""
    counter = Counter(pixel_labels)
    total_pixels = sum(counter.values())
    
    data_rows = []
    for color_name, pixel_count in counter.items():
        if color_name == "unclassified" or color_name is None:
            continue
            
        # Get pixels of this color
        mask = np.array(pixel_labels) == color_name
        color_pixels = flat_pixels[mask]
        
        if len(color_pixels) > 0:
            mean_rgb = np.mean(color_pixels, axis=0)
            std_rgb = np.std(color_pixels, axis=0)
            mean_intensity = np.mean(color_pixels)
            
            data_rows.append({
                "Color_Type": color_name,
                "Pixel_Count": pixel_count,
                "Percentage_Area": round((pixel_count / total_pixels) * 100, 3),
                "Mean_R": round(mean_rgb[0], 2),
                "Mean_G": round(mean_rgb[1], 2),
                "Mean_B": round(mean_rgb[2], 2),
                "Std_R": round(std_rgb[0], 2),
                "Std_G": round(std_rgb[1], 2),
                "Std_B": round(std_rgb[2], 2),
                "Mean_Intensity": round(mean_intensity, 2),
                "Image_Source": image_name
            })
    
    return pd.DataFrame(data_rows).sort_values("Percentage_Area", ascending=False)

def detect_anomalies(pixel_labels, image_shape, anomaly_threshold=1.0):
    """Detect and locate anomalous regions in the image"""
    counter = Counter(pixel_labels)
    total_pixels = sum(counter.values())
    
    rare_colors = []
    for color_name, count in counter.items():
        if color_name and color_name != "unclassified":
            percentage = (count / total_pixels) * 100
            if percentage < anomaly_threshold:
                rare_colors.append((color_name, percentage, count))
    
    return sorted(rare_colors, key=lambda x: x[1])  # Sort by percentage

def create_segmentation_overlay(original_image, pixel_labels, target_color):
    """Create an overlay showing specific color regions"""
    h, w = original_image.shape[:2]
    label_array = np.array(pixel_labels).reshape(h, w)
    
    # Create binary mask for target color
    binary_mask = (label_array == target_color).astype(np.uint8)
    
    # Find connected components
    labeled_regions = label(binary_mask)
    regions = regionprops(labeled_regions)
    
    # Create overlay
    overlay = original_image.copy()
    for region in regions:
        # Get bounding box
        min_row, min_col, max_row, max_col = region.bbox
        
        # Draw rectangle
        cv2.rectangle(overlay, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
        
        # Add label
        cv2.putText(overlay, f"{target_color}", (min_col, min_row-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return overlay, regions

# Streamlit App
def main():
    st.set_page_config(page_title="Medical Image Color Classifier", layout="wide")
    
    st.title("🔬 Advanced Medical Image Color Classification System")
    st.markdown("### Precision color analysis for medical imaging with enhanced green detection")
    
    # Initialize session state
    if 'custom_colors' not in st.session_state:
        st.session_state.custom_colors = {}
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Classification Parameters")
    
    color_threshold = st.sidebar.slider(
        "Color Matching Threshold (Delta E)", 
        min_value=10, max_value=50, value=25, step=1,
        help="Lower values = stricter color matching"
    )
    
    enhance_contrast = st.sidebar.checkbox(
        "Enhance Image Contrast", 
        value=True,
        help="Apply CLAHE contrast enhancement for better color detection"
    )
    
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Detection Threshold (%)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Colors below this percentage are considered anomalies"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Medical Image", 
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        help="Supported formats: JPG, PNG, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption=f"Original: {image_name}", use_column_width=True)
            
            # Preprocess image
            img_array = preprocess_image(image, enhance_contrast)
            
            with col2:
                if enhance_contrast:
                    st.subheader("✨ Enhanced Image")
                    st.image(img_array, caption="Contrast Enhanced", use_column_width=True)
                else:
                    st.subheader("📊 Processing Info")
                    st.info(f"Image Size: {img_array.shape[1]} × {img_array.shape[0]} pixels")
                    st.info(f"Total Pixels: {img_array.shape[0] * img_array.shape[1]:,}")
            
            # Convert reference colors to Lab
            ref_lab_colors = convert_colors_to_lab(HUE_REF_COLORS)
            
            # Add custom colors if any
            if st.session_state.custom_colors:
                ref_lab_colors.update(st.session_state.custom_colors)
            
            # Interactive pixel picking for custom color training
            st.subheader("🎯 Custom Color Training")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                pick_x = st.number_input(
                    "X Coordinate", 
                    min_value=0, max_value=img_array.shape[1]-1, value=0
                )
            
            with col4:
                pick_y = st.number_input(
                    "Y Coordinate", 
                    min_value=0, max_value=img_array.shape[0]-1, value=0
                )
            
            with col5:
                custom_label = st.text_input("Color Label", placeholder="e.g., tissue_type_1")
            
            if st.button("🔍 Add Custom Color Reference"):
                if custom_label:
                    picked_rgb = tuple(img_array[int(pick_y), int(pick_x)])
                    picked_lab = rgb2lab(np.uint8([[picked_rgb]]))[0][0]
                    
                    if custom_label not in st.session_state.custom_colors:
                        st.session_state.custom_colors[custom_label] = []
                    st.session_state.custom_colors[custom_label].append(picked_lab)
                    
                    st.success(f"Added {custom_label} with RGB: {picked_rgb}")
                    st.experimental_rerun()
            
            # Display current custom colors
            if st.session_state.custom_colors:
                st.info(f"Custom colors defined: {', '.join(st.session_state.custom_colors.keys())}")
            
            # Classify pixels
            st.subheader("🔄 Processing Image...")
            
            with st.spinner("Classifying pixels..."):
                h, w = img_array.shape[:2]
                flat_pixels = img_array.reshape(-1, 3)
                lab_pixels = rgb2lab(flat_pixels.reshape(-1, 1, 3)).reshape(-1, 3)
                
                # Classify each pixel
                pixel_labels = []
                classification_distances = []
                
                progress_bar = st.progress(0)
                total_pixels = len(lab_pixels)
                
                for i, lab_pixel in enumerate(lab_pixels):
                    label, distance = classify_pixel_color(
                        lab_pixel, ref_lab_colors, color_threshold
                    )
                    pixel_labels.append(label if label else "unclassified")
                    classification_distances.append(distance if distance != float("inf") else None)
                    
                    # Update progress bar every 1000 pixels
                    if i % 1000 == 0:
                        progress_bar.progress(i / total_pixels)
                
                progress_bar.progress(1.0)
            
            # Create statistics
            stats_df = create_color_statistics(pixel_labels, flat_pixels, image_name)
            
            # Display results
            st.subheader("📊 Color Classification Results")
            
            col6, col7 = st.columns(2)
            
            with col6:
                st.dataframe(stats_df, use_container_width=True)
                
                # Download button
                csv_data = stats_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"{image_name}_color_analysis.csv",
                    mime="text/csv"
                )
            
            with col7:
                # Create pie chart
                if not stats_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors_for_pie = stats_df['Color_Type'].tolist()
                    percentages = stats_df['Percentage_Area'].tolist()
                    
                    ax.pie(percentages, labels=colors_for_pie, autopct='%1.1f%%', startangle=90)
                    ax.set_title("Color Distribution")
                    st.pyplot(fig)
            
            # Reconstructed image
            st.subheader("🎨 Reconstructed Color Map")
            
            # Create reconstructed image
            reconstructed = np.zeros((h * w, 3), dtype=np.uint8)
            for i, color_name in enumerate(pixel_labels):
                if color_name and color_name != "unclassified" and color_name in HUE_REF_COLORS:
                    reconstructed[i] = HUE_REF_COLORS[color_name][0]  # Use first reference color
                else:
                    reconstructed[i] = [128, 128, 128]  # Gray for unclassified
            
            reconstructed_img = reconstructed.reshape((h, w, 3))
            st.image(reconstructed_img, caption="Color-Classified Reconstruction", use_column_width=True)
            
            # Anomaly detection
            st.subheader("🚨 Anomaly Detection")
            
            anomalies = detect_anomalies(pixel_labels, (h, w), anomaly_threshold)
            
            if anomalies:
                st.warning(f"Found {len(anomalies)} rare color types:")
                
                for color_name, percentage, count in anomalies:
                    col8, col9 = st.columns([2, 1])
                    
                    with col8:
                        st.write(f"**{color_name}**: {percentage:.2f}% ({count:,} pixels)")
                    
                    with col9:
                        if st.button(f"Show {color_name} regions", key=f"show_{color_name}"):
                            overlay, regions = create_segmentation_overlay(img_array, pixel_labels, color_name)
                            st.image(overlay, caption=f"Highlighted regions: {color_name}")
                            st.info(f"Found {len(regions)} separate regions of {color_name}")
            else:
                st.success("No anomalies detected with current threshold")
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            total_classified = sum(1 for label in pixel_labels if label and label != "unclassified")
            total_unclassified = sum(1 for label in pixel_labels if not label or label == "unclassified")
            classification_rate = (total_classified / len(pixel_labels)) * 100
            
            col10, col11, col12, col13 = st.columns(4)
            
            with col10:
                st.metric("Total Pixels", f"{len(pixel_labels):,}")
            
            with col11:
                st.metric("Classified Pixels", f"{total_classified:,}")
            
            with col12:
                st.metric("Classification Rate", f"{classification_rate:.1f}%")
            
            with col13:
                st.metric("Unique Colors", len(stats_df))
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a medical image to begin analysis")
        
        # Show color reference
        st.subheader("🎨 Color Reference Guide")
        
        cols = st.columns(3)
        color_names = list(HUE_REF_COLORS.keys())
        
        for i, color_name in enumerate(color_names):
            with cols[i % 3]:
                sample_rgb = HUE_REF_COLORS[color_name][0]
                color_swatch = np.full((50, 100, 3), sample_rgb, dtype=np.uint8)
                st.image(color_swatch, caption=color_name.replace('_', ' ').title())

if __name__ == "__main__":
    main()
