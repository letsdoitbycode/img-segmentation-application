import streamlit as st
from PIL import Image
import numpy as np
import cv2
from algorithms.thresholding import apply_thresholding
from algorithms.kmeans import apply_kmeans
from algorithms.watershed import apply_watershed
from algorithms.canny import apply_canny  # Import Canny edge detection

def main():
    st.title("Image Segmentation App")
    st.markdown("""
    **Unlock the Power of Image Segmentation!**
    Explore and apply different segmentation techniques like Thresholding, K-Means, Watershed, and Canny Edge Detection to analyze and transform your images.
    """)
    
    st.sidebar.header("Segmentation Algorithms")
    options = ["Thresholding", "K-Means", "Watershed", "Canny Edge Detection"]  
    choice = st.sidebar.selectbox("Select an algorithm", options)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  
        image = np.array(image)

        if choice == "Thresholding":
            threshold_method = st.sidebar.radio("Select Thresholding Method", ("global", "adaptive"))
            result = apply_thresholding(image, method=threshold_method)
        elif choice == "K-Means":
            k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)  # Slider for K value
            result = apply_kmeans(image, k)
        elif choice == "Watershed":
            result = apply_watershed(image)
        elif choice == "Canny Edge Detection":
            lower = st.sidebar.slider("Lower Threshold", 0, 255, 100)
            upper = st.sidebar.slider("Upper Threshold", 0, 255, 200)
            result = apply_canny(image, lower, upper)

        st.image(result, caption=f"Result of {choice}", use_container_width=True)  # Updated here

if __name__ == "__main__":
    main()
