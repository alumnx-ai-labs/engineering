from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,  # Higher for creative content
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define State (shared data across all nodes)
class ContentState(TypedDict):
    topic: str
    research_data: str
    outline: str
    draft: str
    final_content: str
    revision_count: int
    quality_score: Annotated[int, operator.add]  # Can be incremented
    needs_revision: bool

# Node 1: Research
def research_node(state: ContentState) -> ContentState:
    """Search web for information on the topic."""
    print(f"\n🔍 RESEARCHING: {state['topic']}")
    
    results = tavily_client.search(state['topic'], max_results=5)
    research_data = "\n\n".join([
        f"Source: {r['title']}\n{r['content']}" 
        for r in results['results']
    ])
    
    return {
        **state,
        "research_data": research_data,
        "quality_score": 20  # Research complete = +20 points
    }

# Node 2: Create Outline
def outline_node(state: ContentState) -> ContentState:
    """Create content outline based on research."""
    print(f"\n📝 CREATING OUTLINE")
    
    prompt = f"""Based on this research about '{state['topic']}':

{state['research_data'][:2000]}

Create a detailed blog post outline with:
- Catchy title
- 4-5 main sections
- Key points for each section
- Estimated word count per section

Format as a structured outline."""

    response = llm.invoke(prompt)
    
    return {
        **state,
        "outline": response.content,
        "quality_score": 20  # Outline complete = +20 points
    }

# Node 3: Write Draft
def draft_node(state: ContentState) -> ContentState:
    """Write first draft based on outline."""
    print(f"\n✍️ WRITING DRAFT (Attempt {state['revision_count'] + 1})")
    
    prompt = f"""Write a complete blog post following this outline:

{state['outline']}

Use this research data:
{state['research_data'][:1500]}

Requirements:
- Engaging introduction
- Well-structured paragraphs
- Use data from research
- Professional tone
- 800-1000 words
- Conclusion with call-to-action"""

    response = llm.invoke(prompt)
    
    return {
        **state,
        "draft": response.content,
        "revision_count": state.get('revision_count', 0) + 1,
        "quality_score": 30  # Draft complete = +30 points
    }

# Node 4: Review Quality
def review_node(state: ContentState) -> ContentState:
    """Review draft and decide if revision needed."""
    print(f"\n🔍 REVIEWING DRAFT")
    
    prompt = f"""Review this blog post draft and provide:

{state['draft']}

1. Quality score (1-10)
2. Issues found (grammar, structure, clarity)
3. Should it be revised? (YES/NO)

Format:
SCORE: X/10
ISSUES: [list issues]
REVISION_NEEDED: YES/NO"""

    response = llm.invoke(prompt)
    review = response.content
    
    # Parse review
    needs_revision = "REVISION_NEEDED: YES" in review
    score_line = [line for line in review.split('\n') if 'SCORE:' in line]
    score = int(score_line[0].split('/')[0].split(':')[1].strip()) if score_line else 7
    
    print(f"   Quality Score: {score}/10")
    print(f"   Needs Revision: {needs_revision}")
    
    return {
        **state,
        "needs_revision": needs_revision and state['revision_count'] < 2,  # Max 2 revisions
        "quality_score": score * 5  # Convert to points
    }

# Node 5: Finalize
def finalize_node(state: ContentState) -> ContentState:
    """Finalize content and add metadata."""
    print(f"\n✅ FINALIZING CONTENT")
    
    final_content = f"""# FINAL BLOG POST

## Topic: {state['topic']}
## Quality Score: {state['quality_score']}/100
## Revisions: {state['revision_count']}

---

{state['draft']}

---

## Metadata
- Word Count: {len(state['draft'].split())}
- Research Sources: 5
- AI-Generated: Yes
"""
    
    return {
        **state,
        "final_content": final_content
    }

# Conditional Edge: Decide if revision needed
def should_revise(state: ContentState) -> str:
    """Route to revision or finalization."""
    if state.get('needs_revision', False):
        print(f"   ⚠️ Routing to REVISION")
        return "revise"
    else:
        print(f"   ✅ Routing to FINALIZE")
        return "finalize"

# Build the Graph
workflow = StateGraph(ContentState)

# Add nodes
workflow.add_node("research", research_node)
workflow.add_node("outline", outline_node)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

# Add edges (workflow flow)
workflow.add_edge("research", "outline")
workflow.add_edge("outline", "draft")
workflow.add_edge("draft", "review")

# Conditional edge: review → revise OR finalize
workflow.add_conditional_edges(
    "review",
    should_revise,
    {
        "revise": "draft",      # Loop back to draft
        "finalize": "finalize"  # Move to finalize
    }
)

workflow.add_edge("finalize", END)

# Set entry point
workflow.set_entry_point("research")

# Compile the graph
app = workflow.compile()

# Visualize the Graph
def visualize_graph():
    """Generate and save graph visualization."""
    try:
        # Generate PNG
        graph_png = app.get_graph().draw_mermaid_png()
        
        # Save to file
        with open("workflow_graph.png", "wb") as f:
            f.write(graph_png)
        
        print("\n📊 Graph visualization saved as 'workflow_graph.png'")
        print("   Open this file to see the workflow structure!\n")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate graph visualization: {e}")
        print("   Install graphviz: brew install graphviz (Mac) or apt-get install graphviz (Linux)")
        print("   Or view graph in LangSmith instead.\n")

# Test
if __name__ == "__main__":
    # Visualize the workflow first
    print("=" * 100)
    visualize_graph()
    print("=" * 100)
    
    topics = [
        "Impact of AI on Software Engineering Jobs in 2025",
        "Best Practices for Machine Learning Model Deployment",
        "Future of Remote Work in Tech Industry"
    ]
    
    for topic in topics:
        print("=" * 100)
        print(f"\n🚀 STARTING WORKFLOW: {topic}\n")
        
        # Initialize state
        initial_state = {
            "topic": topic,
            "research_data": "",
            "outline": "",
            "draft": "",
            "final_content": "",
            "revision_count": 0,
            "quality_score": 0,
            "needs_revision": False
        }
        
        # Run workflow
        final_state = app.invoke(initial_state)
        
        # Print results
        print("\n" + "=" * 100)
        print("📊 WORKFLOW COMPLETE")
        print(f"   Total Quality Score: {final_state['quality_score']}/100")
        print(f"   Total Revisions: {final_state['revision_count']}")
        print("\n📄 FINAL CONTENT:")
        print(final_state['final_content'][:500] + "...")
        print("=" * 100 + "\n")