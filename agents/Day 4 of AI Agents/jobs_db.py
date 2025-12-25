"""Mock job database for testing."""

JOBS_DATABASE = [
    {
        "id": "JOB001",
        "title": "Senior Machine Learning Engineer",
        "company": "TechCorp India",
        "location": "Bangalore",
        "experience": "5-7 years",
        "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS"],
        "description": "Build and deploy ML models at scale. Experience with deep learning and cloud platforms required.",
        "salary": "₹25-35 LPA"
    },
    {
        "id": "JOB002",
        "title": "Data Scientist",
        "company": "FinTech Solutions",
        "location": "Mumbai",
        "experience": "3-5 years",
        "skills": ["Python", "SQL", "Machine Learning", "Statistics", "Tableau"],
        "description": "Analyze financial data and build predictive models. Strong statistics background needed.",
        "salary": "₹18-25 LPA"
    },
    {
        "id": "JOB003",
        "title": "AI Research Engineer",
        "company": "AI Innovations",
        "location": "Hyderabad",
        "experience": "4-6 years",
        "skills": ["Python", "PyTorch", "NLP", "Computer Vision", "Research"],
        "description": "Research and develop cutting-edge AI algorithms. PhD or publications preferred.",
        "salary": "₹30-40 LPA"
    },
    {
        "id": "JOB004",
        "title": "Junior Data Analyst",
        "company": "StartupXYZ",
        "location": "Pune",
        "experience": "1-2 years",
        "skills": ["Python", "SQL", "Excel", "Power BI", "Statistics"],
        "description": "Entry-level position for data analysis and visualization. Great learning opportunity.",
        "salary": "₹6-10 LPA"
    },
    {
        "id": "JOB005",
        "title": "MLOps Engineer",
        "company": "CloudScale Inc",
        "location": "Remote",
        "experience": "3-5 years",
        "skills": ["Docker", "Kubernetes", "Python", "CI/CD", "AWS/Azure"],
        "description": "Deploy and monitor ML models in production. DevOps experience required.",
        "salary": "₹20-28 LPA"
    }
]

def search_jobs(query: str = "", skills: list = None, experience: str = "") -> list:
    """Search jobs based on criteria."""
    results = JOBS_DATABASE
    
    # Filter by skills if provided
    if skills:
        results = [
            job for job in results 
            if any(skill.lower() in [s.lower() for s in job['skills']] for skill in skills)
        ]
    
    # Simple text search in title/description
    if query:
        results = [
            job for job in results
            if query.lower() in job['title'].lower() or query.lower() in job['description'].lower()
        ]
    
    return results

def get_job_by_id(job_id: str) -> dict:
    """Get specific job by ID."""
    for job in JOBS_DATABASE:
        if job['id'] == job_id:
            return job
    return None
