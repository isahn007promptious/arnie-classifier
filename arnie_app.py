import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageDraw
from common import strong_faces, crop_face_from_box, predict_proba_tta

@st.cache_resource
def load_model():
    """Load the trained model, threshold and t_low."""
    model_data = joblib.load("model_out/arnie_model.joblib")
    return model_data["model"], model_data["threshold"], model_data.get("t_low", 0.0), model_data["use_hog"]

st.title("Is this Arnold?")

st.sidebar.markdown("**About this classifier**")
st.sidebar.info("Trained on LFW plus extra photos. Best on front-facing faces.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    pil_image = Image.open(uploaded_file)

    # Detect strong faces
    strong = strong_faces(pil_image)

    if len(strong) == 0:
        st.image(pil_image, width=600)
        st.error("❌ No clear face found. Try a closer, front-facing photo.")
    else:
        # Draw boxes on scored faces only
        display_img = pil_image.copy()
        draw = ImageDraw.Draw(display_img)

        # Load model
        model, threshold, t_low, use_hog = load_model()

        # Pick the best face by detector weight and size (NOT by model score)
        # Sort by: weight (if available) then by box size
        strong_sorted = sorted(strong, key=lambda c: (
            c['weight'] if c['weight'] is not None else 0,
            c['box'][2] * c['box'][3]
        ), reverse=True)

        best_candidate = strong_sorted[0]
        best_box = best_candidate['box']
        best_crop = crop_face_from_box(pil_image, best_box)

        if best_crop is None:
            st.error("❌ Could not process the detected face.")
        else:
            # Score only the best face
            best_proba = predict_proba_tta(model, best_crop, use_hog)

        # Draw boxes around all scored faces
        for cand in strong:
            x, y, w, h = cand['box']
            color = "green" if cand['box'] == best_box else "orange"
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        st.image(display_img, caption="Uploaded Image with Detected Faces", width=600)

        if len(strong) > 1:
            st.info(f"ℹ️ {len(strong)} faces detected. Showing result for the best-scoring face (green box).")

        if best_crop is None:
            st.error("❌ Could not process faces from the image.")
        else:
            # Display cropped face
            st.image(best_crop, caption="Best Scoring Face", width=200, clamp=True)

            # Check if face is small
            box_width = best_box[2]
            if box_width < 60:
                st.warning("⚠️ Face is small or low-resolution, result is less reliable.")

            # Three-state verdict
            if best_proba >= threshold:
                verdict = "arnold"
                confidence = 50 + 50 * (best_proba - threshold) / (1 - threshold)
            elif best_proba >= t_low:
                verdict = "maybe"
                confidence = 50 + 50 * (best_proba - t_low) / (threshold - t_low)
            else:
                verdict = "not_arnold"
                confidence = 50 + 50 * (t_low - best_proba) / t_low

            # Display result
            st.markdown("---")

            if verdict == "arnold":
                st.success(f"✅ **Arnold Schwarzenegger** ({confidence:.1f}% confidence)")
            elif verdict == "maybe":
                st.warning(f"🟡 **Possibly Arnold, not certain** ({confidence:.1f}% confidence)")
            else:
                st.info(f"**Not Arnold** ({confidence:.1f}% confidence)")

            # Additional notes
            if verdict == "maybe":
                st.caption("The model detects some Arnold-like features but isn't confident enough for a definitive match.")



