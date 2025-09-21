import requests

# 1️⃣ Your JWT token from Step 2
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1ODQwMzMzMn0.KTwZs0wP78vEuG9CmXyKNkGn3oFhj010OyrcOyXROu8"  # replace this

# 2️⃣ Headers with the token
headers = {"Authorization": f"Bearer {token}"}

# 3️⃣ The resume file to send
files = {"file": ("resume.pdf", open("resume.pdf", "rb"), "application/pdf")}

# 4️⃣ The job description
data = {"job_description": "Looking for a Python developer with AWS and Docker experience."}

# 5️⃣ Send POST request to FastAPI
response = requests.post(
    "http://127.0.0.1:8000/analyze_resume",  # FastAPI endpoint
    files=files,
    data=data,
    headers=headers
)

# 6️⃣ Print the result
print(response.json())
