# resume_parser_matcher.py

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resume (in real case, you'll read a file or use input)
resume_text = """
Shaista Fatima
Skills: Python, Machine Learning, Data Analysis, SQL, Communication
Experience: 6 months internship at XYZ as Data Analyst
"""

# Sample job description
job_description = """
Looking for a candidate with skills in Python, SQL, Data Analysis, and Communication.
Familiarity with Machine Learning is a bonus.
"""

# Clean text
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

resume_clean = clean_text(resume_text)
job_clean = clean_text(job_description)

# Convert text to vectors
vectorizer = CountVectorizer().fit_transform([resume_clean, job_clean])
vectors = vectorizer.toarray()

# Calculate similarity
similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100

print(f"🔍 Resume matches the job description by {similarity_score:.2f}%")
