from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
import operator
import os
from dotenv import load_dotenv
from jobs_db import search_jobs, get_job_by_id

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define State for Multi-Agent System
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], operator.add]
    resume_text: str
    resume_analysis: str
    matched_jobs: list
    top_job: dict
    cover_letter: str
    next_agent: str

# AGENT 1: Resume Analyzer
def resume_analyzer_agent(state: AgentState) -> AgentState:
    """Specialized agent for analyzing resumes."""
    print("\n📄 RESUME ANALYZER AGENT WORKING...")
    
    system_prompt = """You are an expert resume analyzer with 15 years of HR experience.
Your job is to:
1. Extract key skills, technologies, and tools
2. Identify years of experience and seniority level
3. Highlight strengths and unique qualifications
4. Note any gaps or areas for improvement

Be thorough and structured in your analysis."""

    prompt = f"""{system_prompt}

Analyze this resume:

{state['resume_text']}

Provide a detailed analysis in this format:
SKILLS: [list]
EXPERIENCE: [years/level]
STRENGTHS: [list]
AREAS TO IMPROVE: [list]
RECOMMENDED ROLES: [list]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("   ✓ Resume analyzed")
    
    return {
        **state,
        "resume_analysis": response.content,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"Resume Analyzer: {response.content[:200]}...")
        ],
        "next_agent": "job_matcher"
    }

# AGENT 2: Job Matcher
def job_matcher_agent(state: AgentState) -> AgentState:
    """Specialized agent for matching candidates to jobs."""
    print("\n🔍 JOB MATCHER AGENT WORKING...")
    
    # Extract skills from resume analysis
    analysis = state['resume_analysis']
    
    system_prompt = """You are an expert job matching specialist.
Your job is to:
1. Understand candidate's profile deeply
2. Match them to suitable positions
3. Rank jobs by fit percentage
4. Explain why each job is a good match

Be precise and justify your recommendations."""

    # Search jobs (in real system, this would query your MongoDB)
    all_jobs = search_jobs()
    jobs_text = "\n\n".join([
        f"JOB {i+1} (ID: {job['id']})\n"
        f"Title: {job['title']}\n"
        f"Company: {job['company']}\n"
        f"Skills Required: {', '.join(job['skills'])}\n"
        f"Experience: {job['experience']}\n"
        f"Description: {job['description']}"
        for i, job in enumerate(all_jobs)
    ])
    
    prompt = f"""{system_prompt}

Resume Analysis:
{analysis}

Available Jobs:
{jobs_text}

Select the TOP 3 best-matching jobs and explain why. Format:

MATCH 1: [Job ID] - [Job Title]
FIT SCORE: X/100
REASON: [why this is a good match]

MATCH 2: ...
MATCH 3: ...
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Extract job IDs (simple parsing - in production, use structured output)
    matched_job_ids = []
    for line in response.content.split('\n'):
        if 'MATCH' in line and 'JOB' in line:
            # Extract JOB00X pattern
            for job in all_jobs:
                if job['id'] in line:
                    matched_job_ids.append(job['id'])
                    break
    
    # Get full job details
    matched_jobs = [get_job_by_id(jid) for jid in matched_job_ids if get_job_by_id(jid)]
    
    print(f"   ✓ Matched {len(matched_jobs)} jobs")
    
    return {
        **state,
        "matched_jobs": matched_jobs,
        "top_job": matched_jobs[0] if matched_jobs else None,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"Job Matcher: Found {len(matched_jobs)} matches. {response.content[:200]}...")
        ],
        "next_agent": "cover_letter_writer"
    }

# AGENT 3: Cover Letter Writer
def cover_letter_writer_agent(state: AgentState) -> AgentState:
    """Specialized agent for writing cover letters."""
    print("\n✍️ COVER LETTER WRITER AGENT WORKING...")
    
    if not state['top_job']:
        print("   ✗ No job to write cover letter for")
        return {
            **state,
            "cover_letter": "No suitable jobs found.",
            "next_agent": "END"
        }
    
    system_prompt = """You are an expert cover letter writer with a background in recruiting.
Your job is to:
1. Write compelling, personalized cover letters
2. Highlight relevant experience and skills
3. Show enthusiasm and cultural fit
4. Keep it concise (250-300 words)
5. Use professional yet warm tone

Make each letter unique and authentic."""

    job = state['top_job']
    
    prompt = f"""{system_prompt}

Resume Analysis:
{state['resume_analysis']}

Job Details:
Title: {job['title']}
Company: {job['company']}
Skills Required: {', '.join(job['skills'])}
Description: {job['description']}

Write a compelling cover letter for this position. Address:
- Why this candidate is perfect for this role
- Specific skills that match requirements
- Enthusiasm for the company/role
- Call to action

Format as a professional cover letter.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("   ✓ Cover letter written")
    
    return {
        **state,
        "cover_letter": response.content,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"Cover Letter Writer: Letter completed for {job['title']} at {job['company']}")
        ],
        "next_agent": "END"
    }

# SUPERVISOR AGENT (Orchestrator)
def supervisor_agent(state: AgentState) -> AgentState:
    """Supervisor that coordinates the workflow."""
    print("\n👔 SUPERVISOR AGENT DECIDING...")
    
    next_agent = state.get('next_agent', 'resume_analyzer')
    
    print(f"   → Routing to: {next_agent}")
    
    return {
        **state,
        "next_agent": next_agent
    }

# Routing function for conditional edges
def route_to_agent(state: AgentState) -> str:
    """Route to next agent based on state."""
    next_agent = state.get('next_agent', 'resume_analyzer')
    
    if next_agent == "END":
        return "end"
    
    return next_agent

# Build the Multi-Agent Workflow
workflow = StateGraph(AgentState)

# Add agent nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("resume_analyzer", resume_analyzer_agent)
workflow.add_node("job_matcher", job_matcher_agent)
workflow.add_node("cover_letter_writer", cover_letter_writer_agent)

# Set entry point
workflow.set_entry_point("supervisor")

# Add conditional edges from supervisor to agents
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "resume_analyzer": "resume_analyzer",
        "job_matcher": "job_matcher",
        "cover_letter_writer": "cover_letter_writer",
        "end": END
    }
)

# Each agent returns to supervisor
workflow.add_edge("resume_analyzer", "supervisor")
workflow.add_edge("job_matcher", "supervisor")
workflow.add_edge("cover_letter_writer", "supervisor")

# Compile
app = workflow.compile()

# Visualize
def visualize_graph():
    """Generate and save graph visualization."""
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open("multi_agent_graph.png", "wb") as f:
            f.write(graph_png)
        print("\n📊 Multi-agent graph saved as 'multi_agent_graph.png'\n")
    except Exception as e:
        print(f"\n⚠️ Could not generate graph: {e}")
        print("\n📊 Multi-Agent Structure:")
        print("""
                    Supervisor
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    Resume         Job            Cover Letter
    Analyzer       Matcher        Writer
        │               │               │
        └───────────────┴───────────────┘
                        │
                   Supervisor
        """)

# Test with sample resumes
if __name__ == "__main__":
    visualize_graph()
    
    # Sample resume
    sample_resume = """
RAJESH KUMAR
Senior Software Engineer
Email: rajesh.kumar@email.com | Phone: +91-9876543210
LinkedIn: linkedin.com/in/rajeshkumar

EXPERIENCE:
Software Engineer at TechStartup (2019-2024)
- Built ML models for recommendation systems using Python and TensorFlow
- Deployed models to AWS using Docker and Kubernetes
- Improved model accuracy by 25% through feature engineering
- Led team of 3 junior engineers

Junior Developer at InfoSys (2017-2019)
- Developed data pipelines using Python and SQL
- Created dashboards using Tableau and Power BI

SKILLS:
- Languages: Python, SQL, Java
- ML/AI: TensorFlow, PyTorch, Scikit-learn
- Cloud: AWS (EC2, S3, SageMaker)
- Tools: Docker, Git, Jenkins
- Data: Pandas, NumPy, Spark

EDUCATION:
B.Tech in Computer Science, IIT Madras (2013-2017)

CERTIFICATIONS:
- AWS Certified Machine Learning Specialty
- TensorFlow Developer Certificate
"""

    print("=" * 100)
    print("🚀 STARTING MULTI-AGENT JOB APPLICATION ASSISTANT")
    print("=" * 100)
    
    # Initialize state
    initial_state = {
        "messages": [],
        "resume_text": sample_resume,
        "resume_analysis": "",
        "matched_jobs": [],
        "top_job": None,
        "cover_letter": "",
        "next_agent": "resume_analyzer"
    }
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 100)
    print("📊 MULTI-AGENT WORKFLOW COMPLETE")
    print("=" * 100)
    
    print("\n📄 RESUME ANALYSIS:")
    print(final_state['resume_analysis'])
    
    print("\n\n🎯 MATCHED JOBS:")
    for i, job in enumerate(final_state['matched_jobs'], 1):
        print(f"\n{i}. {job['title']} at {job['company']}")
        print(f"   Location: {job['location']} | Salary: {job['salary']}")
        print(f"   Skills: {', '.join(job['skills'])}")
    
    print("\n\n✉️ COVER LETTER (for top match):")
    print(final_state['cover_letter'])
    
    print("\n" + "=" * 100)
