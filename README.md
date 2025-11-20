# Smart_resume-reviewer
winter of code social

the complete Smart Resume Reviewer condensed into a single Python file (app.py).

This version keeps all the advanced features (BERT Semantic Analysis, Layout-Aware Parsing) but removes the complexity of setting up a separate backend server.

1. The Code (app.py)
Save the following code into a file named app.py.

Python

import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
code:

import io

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# --- 2. CACHED RESOURCE LOADING ---
# We use @st.cache_resource so the AI model only loads once, making the app fast.
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 3. HELPER FUNCTIONS ---

def extract_text_from_pdf(file_bytes):
    """
    Extracts text using pdfplumber (better for multi-column layouts).
    """
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def analyze_resume(resume_text, jd_text):
    """
    Performs Semantic Analysis (BERT) and Keyword extraction.
    """
    # A. Semantic Similarity Score (The "Brain")
    embeddings1 = model.encode(resume_text, convert_to_tensor=True)
    embeddings2 = model.encode(jd_text, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    match_score = float(cosine_score.item()) * 100

    # B. Basic Keyword Gap Analysis
    # (Simple set logic to find words in JD that are missing from Resume)
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    
    # Filter out short common words (stop words)
    missing_words = [w for w in (jd_words - resume_words) if len(w) > 5]
    
    return match_score, missing_words[:10] # Return top 10 missing

# --- 4. MAIN UI LAYOUT ---

st.title("🚀 Single-File AI Resume Reviewer")
st.markdown("### Powered by BERT & PDFPlumber")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    st.subheader("2. Job Description")
    jd_text = st.text_area("Paste JD here...", height=250)

# Button to trigger analysis
if st.button("Analyze Resume", type="primary"):
    if uploaded_file and jd_text:
        with st.spinner("Reading PDF & crunching numbers with AI..."):
            try:
                # 1. Extract Text
                file_bytes = uploaded_file.read()
                resume_content = extract_text_from_pdf(file_bytes)
                
                # 2. Run Analysis
                score, missing_keywords = analyze_resume(resume_content, jd_text)
                
                # --- DISPLAY RESULTS ---
                st.divider()
                st.header("📊 Analysis Results")
                
                # Gauge / Progress Bar
                score_color = "green" if score > 70 else "orange" if score > 50 else "red"
                st.markdown(f"### Match Score: :{score_color}[{score:.2f}%]")
                st.progress(int(score))

                # Result Columns
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    st.warning("⚠️ Missing Keywords (Top 10)")
                    for word in missing_keywords:
                        st.write(f"- {word}")
                
                with r_col2:
                    st.info("📄 Resume Content Preview")
                    st.text(resume_content[:500] + "...")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a resume and paste a job description first.")
        
2. How it Works (The Flow)
Instead of sending data over the internet to a separate API, this script:

Ingests the PDF directly in memory.

Uses the pre-loaded AI model (loaded once at startup) to create "embeddings" (math representations of the text).

Calculates the similarity and updates the UI instantly.

3. How to Run It
You need to install the libraries first. Open your terminal and run:

to install in the system

pip install streamlit pdfplumber sentence-transformers torch
Then, run the app:

to install in the system

streamlit run app.py
