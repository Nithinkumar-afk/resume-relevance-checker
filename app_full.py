# app_full.py -- Modern Dark-Themed Resume Relevance Checker

import streamlit as st
import pdfplumber, docx, io, os, tempfile, csv, zipfile
from sentence_transformers import SentenceTransformer, util
import spacy, re, json, uuid
from collections import Counter
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import globals
import warnings
import openai  # Required for LLM

# -----------------------
# Suppress Warnings & Fix LangChain verbose
# -----------------------
warnings.filterwarnings("ignore", category=UserWarning)
globals.set_verbose(False)

# -----------------------
# Page Config & Dark Theme
# -----------------------
st.set_page_config(page_title="Resume Relevance Checker", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #1e0f3b, #2a1a5c); color: white; font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;}
[data-testid="stSidebar"] { background-color: #1b0d35; padding: 1rem; }
.resume-card { background-color: rgba(255,255,255,0.05); padding:1rem; border-radius:12px; margin-bottom:1rem; box-shadow:0 4px 6px rgba(0,0,0,0.4);}
.summary-card { background-color: rgba(255,255,255,0.08); padding:1rem; border-radius:12px; margin-bottom:1rem; box-shadow:0 4px 6px rgba(0,0,0,0.5);}
.subtitle { color: #d0c8ff; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.title("Resume Checker Options")
mode = st.sidebar.radio("Select mode:", ["Single resume", "Batch (zip)"])
if mode=="Single resume":
    uploaded_file = st.sidebar.file_uploader("Upload your resume", type=["pdf","docx"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload a ZIP of resumes", type=["zip"])
job_description = st.sidebar.text_area("Job Description", height=120)
job_title = st.sidebar.text_input("Job Title")
company_name = st.sidebar.text_input("Company Name")
show_best_sentences = st.sidebar.checkbox("Show best matching sentences")
skill_strictness = st.sidebar.slider("Skill Matching Strictness",0,100,50)
text_model = st.sidebar.selectbox("Text Comparison Model",["Fast Comparison Model","Semantic Model"])
top_k_sentences = st.sidebar.number_input("Top K sentences",1,20,5)
analyze_button = st.sidebar.button("Analyze & Match")

# -----------------------
# Main Header
# -----------------------
st.markdown("## ✨ Resume Relevance Checker ✨")
st.markdown("<div class='subtitle'>Analyze resumes against your job description.</div>", unsafe_allow_html=True)

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_embedding_model(): return SentenceTransformer("all-mpnet-base-v2")
model = load_embedding_model()

@st.cache_resource
def load_spacy_model():
    try: return spacy.load("en_core_web_sm")
    except: st.error("Run: python -m spacy download en_core_web_sm"); raise
nlp = load_spacy_model()

# LLM Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

jd_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""You are an expert recruiter AI. Extract mandatory skills, optional skills, and certifications from this job description in JSON format.
Job Description:
{job_description}"""
)
resume_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""Analyze the resume text and return JSON with:
- skills_found
- certifications_found
- achievements_count
- readability_score
Resume Text:
{resume_text}"""
)
feedback_prompt = PromptTemplate(
    input_variables=["jd_parsed","resume_analysis"],
    template="""Given job description JSON: {jd_parsed} and resume analysis JSON: {resume_analysis},
generate actionable improvement feedback for the candidate."""
)

# -----------------------
# Helper Functions
# -----------------------
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file) as pdf: return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif file.name.lower().endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def normalize_text(text):
    text = re.sub(r'\r\n?','\n',text)
    text = re.sub(r'\n{2,}','\n',text)
    text = re.sub(r'\s+',' ',text)
    return text.strip()

def semantic_score(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def parse_job_description_with_llm(job_text):
    prompt_text = jd_prompt.format(job_description=job_text)
    try:
        response = llm.generate([prompt_text])
        return json.loads(response.generations[0][0].text)
    except: return {"mandatory_skills":[],"optional_skills":[],"certifications":[]}

def analyze_resume_with_llm(resume_text):
    prompt_text = resume_prompt.format(resume_text=resume_text)
    try:
        response = llm.generate([prompt_text])
        return json.loads(response.generations[0][0].text)
    except: return {"skills_found":[],"certifications_found":[],"achievements_count":0,"readability_score":0}

def generate_feedback_with_llm(jd_parsed,resume_analysis):
    prompt_text = feedback_prompt.format(jd_parsed=jd_parsed,resume_analysis=resume_analysis)
    try:
        response = llm.generate([prompt_text])
        return response.generations[0][0].text
    except: return "Feedback generation failed."

# -----------------------
# Analyze & Display
# -----------------------
if analyze_button and job_description and uploaded_file:
    job_parsed = parse_job_description_with_llm(job_description)
    job_skills = set(job_parsed.get("mandatory_skills",[])) | set(job_parsed.get("optional_skills",[]))

    resumes_data = []

    if mode=="Single resume":
        files_to_process = [uploaded_file]
    else:
        with zipfile.ZipFile(uploaded_file) as z:
            files_to_process = [z.open(f) for f in z.namelist() if f.lower().endswith((".pdf",".docx"))]

    for file in files_to_process:
        if isinstance(file, io.BytesIO):
            content = file.read()
        else:
            content = extract_text(file)
        content_norm = normalize_text(content)

        resume_analysis = analyze_resume_with_llm(content_norm)
        feedback = generate_feedback_with_llm(job_parsed, resume_analysis)
        final_score = 0.5*semantic_score(content_norm,job_description) + 0.5*resume_analysis.get("readability_score",0)
        verdict = "PASS" if final_score>0.6 else "REVIEW"

        resumes_data.append({
            "resume_name": getattr(file,"name",str(file)),
            "final_score": final_score,
            "verdict": verdict,
            "resume_analysis": resume_analysis,
            "feedback": feedback
        })

    # Display Resume Cards
    for resume in resumes_data:
        st.markdown(f"""
        <div class="resume-card">
            <h4>{resume['resume_name']}</h4>
            <p><strong>Job Title:</strong> {job_title} &nbsp; | &nbsp;
            <strong>Company:</strong> {company_name}</p>
            <p><strong>Final Score:</strong> {resume['final_score']*100:.1f}% &nbsp; | &nbsp;
            <strong>Verdict:</strong> {resume['verdict']}</p>
            <p><strong>Skills Found:</strong> {', '.join(resume['resume_analysis']['skills_found'])}</p>
            <p><strong>Certifications Found:</strong> {', '.join(resume['resume_analysis']['certifications_found'])}</p>
            <p><strong>Achievements Count:</strong> {resume['resume_analysis']['achievements_count']}</p>
        </div>
        """, unsafe_allow_html=True)

        if show_best_sentences:
            for i,sentence in enumerate(resume['resume_analysis'].get("skills_found",[])[:top_k_sentences]):
                st.markdown(f"<div class='resume-card' style='padding:0.5rem; margin-left:1rem;'>{i+1}. {sentence}</div>",unsafe_allow_html=True)

    # Overall Summary Card
    if resumes_data:
        st.markdown(f"""
        <div class="summary-card">
            <h4>Overall Match Summary</h4>
            <p><strong>Semantic Match:</strong> {resumes_data[0]['final_score']*100:.1f}%</p>
            <p><strong>Readability:</strong> {resumes_data[0]['resume_analysis']['readability_score']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # CSV Download
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(tmp_csv)
    writer.writerow(["Resume Name","Final Score","Skills Found","Certifications Found","Achievements Count","Feedback"])
    for r in resumes_data:
        writer.writerow([
            r['resume_name'],
            f"{r['final_score']*100:.1f}",
            ", ".join(r['resume_analysis']['skills_found']),
            ", ".join(r['resume_analysis']['certifications_found']),
            r['resume_analysis']['achievements_count'],
            r['feedback']
        ])
    tmp_csv.close()
    st.download_button("Download CSV of Analysis", data=open(tmp_csv.name,"rb").read(), file_name="resume_analysis.csv", mime="text/csv")
