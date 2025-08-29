import streamlit as st
import re
from transformers import pipeline
from PIL import Image

# Load model (for demo: sentiment analysis, can replace with relevancy classifier)
classifier = pipeline("sentiment-analysis")

# Fake policy checks
def check_policy(text):
    reasons = []
    if "http" in text or "www" in text:
        reasons.append("Advertisement (URL detected)")
    if len(text.split()) < 5:
        reasons.append("Too short → Possible spam/rant")
    return reasons

st.title("🚀 Smart Review Quality Checker")

# Text input
review_text = st.text_area("Paste your review text:")

# Image upload
uploaded_image = st.file_uploader("Upload a review image (optional)", type=["jpg","png","jpeg"])
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Review Image")

if st.button("Analyze Review"):
    if review_text:
        # Run ML model
        sentiment = classifier(review_text)[0]
        policies = check_policy(review_text)

        st.subheader("📊 Results")
        st.write(f"**Sentiment:** {sentiment['label']} (score={sentiment['score']:.2f})")

        if policies:
            st.error(f"🚨 Violations: {', '.join(policies)}")
        else:
            st.success("✅ Review Passed: No issues found")
    else:
        st.warning("Please enter some review text.")
