from flask import Flask, render_template, request, jsonify
import spacy
import os
from pdfminer.high_level import extract_text
from docx import Document
import io
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_bytes
import pytesseract
from PyPDF2 import PdfReader
import logging
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Explicitly set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# Uncomment and adjust for macOS/Linux if needed:
# pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"  # macOS
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Static company data
COMPANY_DATA = {
    "company1": {
        "name": "CISCO",
        "info": "A leading tech firm specializing in software solutions.",
        "eligibility": "B.Tech/B.E. with 60%+ marks, no active backlogs.",
        "skill_set": ["Python", "Java", "SQL"],
        "roles_and_packages": "Software Engineer (5-7 LPA), Data Analyst (6-8 LPA)",
        "selection_process": "Online Test, Technical Interview, HR Interview",
        "personal_skills": ["Communication", "Teamwork", "Problem Solving"]
    },
    "company2": {
        "name": "Tech Mahindra",
        "info": "Innovative startup focused on AI and machine learning.",
        "eligibility": "B.Tech/M.Tech with 65%+ marks.",
        "skill_set": ["Machine Learning", "TensorFlow", "Python"],
        "roles_and_packages": "AI Engineer (8-10 LPA), Research Analyst (7-9 LPA)",
        "selection_process": "Coding Test, Technical Rounds (2), HR Round",
        "personal_skills": ["Creativity", "Analytical Thinking", "Adaptability"]
    },
    "company3": {
        "name": "SERVICE NOW",
        "info": "Global IT services provider.",
        "eligibility": "Graduate with 60%+ marks.",
        "skill_set": ["C++", "Networking", "Cloud Computing"],
        "roles_and_packages": "Network Engineer (6-8 LPA), Cloud Specialist (7-9 LPA)",
        "selection_process": "Aptitude Test, Technical Interview, Final Interview",
        "personal_skills": ["Leadership", "Time Management", "Collaboration"]
    },
    "springrole": {
        "name": "SpringRole",
        "info": "Blockchain-based professional networking platform.",
        "eligibility": "B.Tech with 65%+ marks, blockchain interest.",
        "skill_set": ["Blockchain", "Solidity", "JavaScript"],
        "roles_and_packages": "Blockchain Developer (7-10 LPA)",
        "selection_process": "Coding Challenge, Technical Interview, HR",
        "personal_skills": ["Innovation", "Communication", "Initiative"]
    },
    "capgemini": {
        "name": "Capgemini",
        "info": "Global leader in consulting, technology, and outsourcing.",
        "eligibility": "B.Tech/B.E./MCA with 60%+ marks.",
        "skill_set": ["Java", "SAP", "Testing"],
        "roles_and_packages": "Software Engineer (4-6 LPA), Consultant (6-8 LPA)",
        "selection_process": "Aptitude Test, Technical Interview, HR Interview",
        "personal_skills": ["Teamwork", "Adaptability", "Problem Solving"]
    },
    "company6": {
        "name": "TCL",
        "info": "Tech firm specializing in cybersecurity.",
        "eligibility": "B.Tech with 60%+ marks.",
        "skill_set": ["Cybersecurity", "Ethical Hacking", "Linux"],
        "roles_and_packages": "Security Analyst (6-8 LPA)",
        "selection_process": "Online Test, Technical Interview, HR",
        "personal_skills": ["Attention to Detail", "Analytical Skills", "Ethics"]
    },
    "microsoft": {
        "name": "Microsoft",
        "info": "Global tech giant in software and cloud services.",
        "eligibility": "B.Tech/M.Tech with 70%+ marks.",
        "skill_set": ["C#", "Azure", "AI"],
        "roles_and_packages": "Software Engineer (12-15 LPA), Cloud Engineer (14-18 LPA)",
        "selection_process": "Coding Test, Multiple Technical Rounds, HR",
        "personal_skills": ["Innovation", "Leadership", "Communication"]
    },
    "wingify": {
        "name": "Wingify",
        "info": "SaaS company focused on conversion optimization.",
        "eligibility": "B.Tech with 65%+ marks.",
        "skill_set": ["JavaScript", "React", "Analytics"],
        "roles_and_packages": "Frontend Developer (8-10 LPA)",
        "selection_process": "Coding Test, Technical Interview, HR",
        "personal_skills": ["Creativity", "Teamwork", "Problem Solving"]
    },
    "accenture": {
        "name": "Accenture",
        "info": "Global professional services company.",
        "eligibility": "B.Tech/B.E. with 60%+ marks.",
        "skill_set": ["Java", "Cloud", "Consulting"],
        "roles_and_packages": "Associate Software Engineer (4-6 LPA), Analyst (5-7 LPA)",
        "selection_process": "Aptitude Test, Technical Interview, HR",
        "personal_skills": ["Adaptability", "Communication", "Collaboration"]
    },
    "cognizant": {
        "name": "Cognizant",
        "info": "IT services and consulting firm.",
        "eligibility": "B.Tech with 60%+ marks.",
        "skill_set": ["Python", "SQL", "DevOps"],
        "roles_and_packages": "Programmer Analyst (4-6 LPA)",
        "selection_process": "Online Test, Technical Interview, HR",
        "personal_skills": ["Teamwork", "Time Management", "Analytical Thinking"]
    },
    "oracle": {
        "name": "Oracle",
        "info": "Leader in database software and cloud solutions.",
        "eligibility": "B.Tech/M.Tech with 65%+ marks.",
        "skill_set": ["SQL", "Java", "Cloud"],
        "roles_and_packages": "Database Engineer (8-12 LPA)",
        "selection_process": "Aptitude Test, Technical Rounds, HR",
        "personal_skills": ["Problem Solving", "Leadership", "Communication"]
    },
    "wipro": {
        "name": "Wipro",
        "info": "Global IT, consulting, and business process services.",
        "eligibility": "B.Tech with 60%+ marks.",
        "skill_set": ["Java", "Testing", "Support"],
        "roles_and_packages": "Project Engineer (3.5-5 LPA)",
        "selection_process": "Online Test, Technical Interview, HR",
        "personal_skills": ["Adaptability", "Teamwork", "Initiative"]
    }
}

def extract_text_from_pdf(file):
    """Extract text from a PDF file using pdfminer, with OCR fallback if needed."""
    try:
        pdf_bytes = file.read()
        pdf_file = io.BytesIO(pdf_bytes)
        if is_encrypted(pdf_file):
            raise ValueError("The PDF is password-protected. Please provide an unprotected file.")
        
        # Try pdfminer first
        text = extract_text(pdf_file)
        if text.strip():
            logging.info(f"Extracted text with pdfminer: {text[:200]}...")  # Increased to 200 chars for more context
            return text.strip()
        
        # Fallback to OCR with Tesseract
        logging.info("No text extracted with pdfminer, attempting OCR with Tesseract")
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=5)  # Limit to 5 pages
        ocr_text = "\n".join([pytesseract.image_to_string(img, config='--psm 3') for img in images])  # Changed to --psm 3 for better table/column detection
        logging.info(f"Extracted text with OCR: {ocr_text[:200]}...")
        if ocr_text.strip():
            logging.info("Text successfully extracted with OCR")
            return ocr_text.strip()
        
        raise ValueError("No text extracted from the file. Ensure the PDF contains readable text or is not a scanned image without OCR support.")
    
    except Exception as e:
        logging.error(f"PDF Extraction Error: {str(e)}")
        raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
def is_encrypted(pdf_file):
    """Check if the PDF is encrypted."""
    try:
        reader = PdfReader(pdf_file)
        return reader.is_encrypted
    except Exception as e:
        logging.error(f"Error checking PDF encryption: {str(e)}")
        return False  # Assume not encrypted if check fails

def extract_text_from_docx(docx_bytes):
    """Extract text from a DOCX file."""
    try:
        doc = Document(io.BytesIO(docx_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        if text.strip():
            logging.info("Text successfully extracted from DOCX")
            return text
        raise ValueError("No text found in DOCX file.")
    except Exception as e:
        logging.error(f"DOCX Extraction Error: {str(e)}")
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")

from fuzzywuzzy import fuzz  # Install: pip install fuzzywuzzy

def extract_resume_data(resume_text):
    """Extract key sections from the resume using regex and spaCy."""
    doc = nlp(resume_text.lower())
    
    # Eligibility (CGPA/Marks) Extraction
    eligibility = []
    edu_patterns = [
        r"(b\.tech|b\.e\.|m\.tech|mca|graduate|bachelor|master)\s*(?:in|of)?\s*([\w\s]+)?\s*(\d+\.?\d*\s*(?:/|out of)?\s*\d+\.?\d*\s*(%|percent|marks|cgpa)?)",  # e.g., "B.Tech 8.5/10 CGPA"
        r"(b\.tech|b\.e\.|m\.tech|mca|graduate|bachelor|master)\s*[-,:;]?\s*(\d+\.?\d*\s*(?:/|out of)?\s*\d+\.?\d*\s*(%|percent|marks|cgpa)?)",  # e.g., "B.Tech: 75%"
        r"(degree|ug|pg)\s*([\w\s]+)?\s*(\d+\.?\d*\s*(?:/|out of)?\s*\d+\.?\d*\s*(%|percent|marks|cgpa)?)",  # e.g., "UG 85%"
        r"(cgpa|marks|percentage)[\s:]*(\d+\.?\d*\s*(?:/|out of)?\s*\d+\.?\d*\s*(%|percent)?)"  # e.g., "CGPA: 7.8/10"
    ]
    for pattern in edu_patterns:
        matches = re.findall(pattern, resume_text.lower())
        for match in matches:
            eligibility.append(" ".join(filter(None, match)))
    logging.info(f"Extracted eligibility: {eligibility}")

    # Technical Skills Extraction
    skills = set()
    skill_keywords = set(keyword.lower() for keyword in sum([data["skill_set"] for data in COMPANY_DATA.values()], []))
    
    # spaCy token matching
    for token in doc:
        token_text = token.text.strip()
        if token_text in skill_keywords:
            skills.add(token_text)
    
    # Regex fallback for skills section
    skills_section = re.search(r"(skills|technical skills|skill set)[\s\S]*?(?=(projects|education|experience|$))", resume_text.lower(), re.IGNORECASE)
    if skills_section:
        section_text = skills_section.group(0)
        potential_skills = re.split(r'[,\n;]+', section_text)
        for skill in potential_skills:
            cleaned_skill = skill.strip().replace("skills:", "").strip()
            # Exact match
            if cleaned_skill in skill_keywords:
                skills.add(cleaned_skill)
            # Fuzzy match for OCR errors
            else:
                for keyword in skill_keywords:
                    if fuzz.ratio(cleaned_skill, keyword) > 80:  # 80% similarity
                        skills.add(keyword)
                        break
    
    # Personal Skills Extraction
    personal_skills = set()
    personal_skill_keywords = set(keyword.lower() for keyword in sum([data["personal_skills"] for data in COMPANY_DATA.values()], []))
    
    # spaCy token matching
    for token in doc:
        token_text = token.text.strip()
        if token_text in personal_skill_keywords:
            personal_skills.add(token_text)
    
    # Regex fallback for personal skills (e.g., "Soft Skills", "Personal Attributes")
    personal_section = re.search(r"(soft skills|personal skills|attributes|strengths)[\s\S]*?(?=(skills|projects|education|experience|$))", resume_text.lower(), re.IGNORECASE)
    if personal_section:
        section_text = personal_section.group(0)
        potential_personal = re.split(r'[,\n;]+', section_text)
        for skill in potential_personal:
            cleaned_skill = skill.strip().replace("soft skills:", "").replace("personal skills:", "").strip()
            if cleaned_skill in personal_skill_keywords:
                personal_skills.add(cleaned_skill)
            else:
                for keyword in personal_skill_keywords:
                    if fuzz.ratio(cleaned_skill, keyword) > 80:
                        personal_skills.add(keyword)
                        break
    
    # Projects and Activities (unchanged)
    projects = []
    project_section = re.search(r"(projects|experience)[\s\S]*?(?=(skills|education|activities|$))", resume_text.lower(), re.IGNORECASE)
    if project_section:
        project_text = project_section.group(0)
        projects = [line.strip() for line in project_text.split("\n") if line.strip() and not any(x in line.lower() for x in ["skills", "education", "activities"])]

    activities = []
    activity_section = re.search(r"(activities|extracurricular)[\s\S]*?(?=(skills|education|projects|$))", resume_text.lower(), re.IGNORECASE)
    if activity_section:
        activity_text = activity_section.group(0)
        activities = [line.strip() for line in activity_text.split("\n") if line.strip() and not any(x in line.lower() for x in ["skills", "education", "projects"])]

    logging.info(f"Extracted skills: {skills}, Personal skills: {personal_skills}")
    return {
        "eligibility": eligibility if eligibility else ["Not specified"],
        "skill_set": list(skills) if skills else ["None detected"],
        "personal_skills": list(personal_skills) if personal_skills else ["None detected"],
        "projects": projects if projects else ["None listed"],
        "activities": activities if activities else ["None listed"],
        "raw_text": resume_text
    }

def compute_tf_idf_similarity(resume_text, job_description):
    """Compute cosine similarity between resume and job description using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        logging.error(f"TF-IDF Similarity Error: {str(e)}")
        return 0.0  # Return 0 if similarity computation fails

def calculate_score(resume_data, company_data, similarity_score):
    """Calculate a compatibility score combining rule-based and TF-IDF similarity."""
    score = 0
    max_score = 100

    resume_edu = " ".join(resume_data["eligibility"]).lower()
    company_edu = company_data["eligibility"].lower()

    degrees = [("b.tech", 15), ("m.tech", 10), ("mca", 5), ("b.e.", 15), ("graduate", 5), ("bachelor", 15)]
    for degree, points in degrees:
        if degree in resume_edu and degree in company_edu:
            score += points
            break

    resume_percent_match = re.search(r"\d+\.?\d*\s*(?:/|out of)?\s*\d+\.?\d*\s*(%|percent|marks|cgpa)?", resume_edu)
    company_percent_match = re.search(r"\d+\.?\d*\s*%", company_edu)
    if resume_percent_match and company_percent_match:
        resume_value = re.search(r"\d+\.?\d*", resume_percent_match.group()).group()
        resume_scale = re.search(r"(?:/|out of)?\s*(\d+\.?\d*)", resume_percent_match.group())
        company_percent = float(company_percent_match.group().replace("%", ""))

        if resume_scale:
            resume_cgpa = float(resume_value)
            max_cgpa = float(resume_scale.group(1))
            resume_percent = (resume_cgpa / max_cgpa) * 100
        else:
            resume_percent = float(resume_value)

        if resume_percent >= company_percent:
            score += 15

    if "no active backlogs" in company_edu and "backlog" not in resume_edu:
        score += 5

    resume_skills = set(skill.lower() for skill in resume_data["skill_set"])  # Ensure lowercase
    company_skills = set(skill.lower() for skill in company_data["skill_set"])  # Ensure lowercase
    skill_matches = len(resume_skills.intersection(company_skills))
    logging.info(f"Resume Skills: {resume_skills}, Company Skills: {company_skills}, Matches: {skill_matches}")
    score += min(skill_matches * 10, 30)

    resume_personal = set(resume_data["personal_skills"])
    company_personal = set(company_data["personal_skills"])
    personal_matches = len(resume_personal.intersection(company_personal))
    score += min(personal_matches * 10, 20)

    if len(resume_data["projects"]) > 1:
        score += 10
    elif len(resume_data["projects"]) == 1:
        score += 5

    tfidf_score = similarity_score * 30
    score += tfidf_score

    return min(score, max_score)

@app.route('/')
@app.route('/home')
def home():
    return render_template('ra1.html')

@app.route('/upload_page')
def upload_page():
    return render_template('ra2.html')

@app.route('/company-details', methods=['GET'])
def company_details():
    company = request.args.get('company', 'microsoft')
    data = COMPANY_DATA.get(company, COMPANY_DATA["microsoft"])
    return render_template('company-details.html', company_data=data)

@app.route('/resume_analyzer', methods=['GET', 'POST'])
def resume_analyzer():
    if request.method == 'POST':
        if "file" not in request.files or not request.files["file"].filename:
            return render_template('resume_analyzer.html', error="No file uploaded!", companies=COMPANY_DATA)

        file = request.files["file"]
        company_key = request.form.get("company", "microsoft")

        try:
            if file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file)
            elif file.filename.endswith(".docx"):
                resume_text = extract_text_from_docx(file.read())
            else:
                return render_template('resume_analyzer.html', error="Unsupported file format. Upload a .pdf or .docx", companies=COMPANY_DATA)

            if not resume_text.strip():
                return render_template('resume_analyzer.html', error="No text extracted from the file. Ensure the PDF/DOCX contains readable text.", companies=COMPANY_DATA)

            resume_data = extract_resume_data(resume_text)
            company_data = COMPANY_DATA.get(company_key, COMPANY_DATA["microsoft"])
            job_description = f"{company_data['info']} Required skills: {', '.join(company_data['skill_set'])}. Eligibility: {company_data['eligibility']}. Roles: {company_data['roles_and_packages']}."

            similarity_score = compute_tf_idf_similarity(resume_data["raw_text"], job_description)
            score = calculate_score(resume_data, company_data, similarity_score)

            return render_template('resume_analyzer.html',
                                  resume_data=resume_data,
                                  company_data=company_data,
                                  similarity_score=similarity_score,
                                  score=score,
                                  companies=COMPANY_DATA)

        except ValueError as e:
            return render_template('resume_analyzer.html', error=str(e), companies=COMPANY_DATA)
        except Exception as e:
            logging.error(f"Unexpected error in resume_analyzer: {str(e)}")
            return render_template('resume_analyzer.html', error=f"An unexpected error occurred: {str(e)}", companies=COMPANY_DATA)

    return render_template('resume_analyzer.html', companies=COMPANY_DATA)


if __name__ == "__main__":
    app.run(debug=True)
