from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
import wikipedia
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Changed from gemini-2.5-flash
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Tool 1: Web Search (for current data)
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, statistics, GDP data, and recent data."""
    results = tavily_client.search(query, max_results=5)
    formatted = []
    for r in results['results']:
        formatted.append(f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}\n")
    return "\n---\n".join(formatted)

# Tool 2: Wikipedia (for background info)
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for general knowledge, historical facts, country information, and background."""
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        page = wikipedia.page(query, auto_suggest=True)
        return f"Summary: {summary}\n\nURL: {page.url}"
    except Exception as e:
        return f"Error: {str(e)}. Try a more specific search term."

# Tool 3: Calculator
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input must be a valid Python expression.
    Examples: '1500000 - 1200000', '(50 + 30) * 2', '100 / 3'"""
    try:
        # Handle common formats
        expression = expression.replace(",", "").replace("$", "").replace("₹", "")
        result = eval(expression)
        return f"Result: {result:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}. Make sure expression is valid Python math."

# Tool 4: Unit Converter
@tool
def convert_units(query: str) -> str:
    """Convert between common units. 
    Format: 'value from_unit to to_unit'
    Examples: '100 USD to INR', '5 kilometers to miles', '1000 million to billion'"""
    try:
        conversions = {
            "million_to_billion": 0.001,
            "billion_to_trillion": 0.001,
            "trillion_to_billion": 1000,
            "billion_to_million": 1000,
            "usd_to_inr": 83,
            "inr_to_usd": 0.012,
            "kilometers_to_miles": 0.621371,
            "miles_to_kilometers": 1.60934
        }
        
        parts = query.lower().split()
        if len(parts) >= 4:
            value = float(parts[0].replace(",", ""))
            from_unit = parts[1]
            to_unit = parts[3]
            key = f"{from_unit}_to_{to_unit}"
            
            if key in conversions:
                result = value * conversions[key]
                return f"{value:,.2f} {from_unit} = {result:,.2f} {to_unit}"
        
        return "Format: 'value from_unit to to_unit'. Example: '100 USD to INR'"
    except Exception as e:
        return f"Error: {str(e)}"

# Create tools list
tools = [web_search, wikipedia_search, calculator, convert_units]

# Create agent with LangGraph
agent_executor = create_react_agent(llm, tools)

# Complex test queries
if __name__ == "__main__":
    queries = [
        "Which cities did Messi visit on his tour in India in December,2025 please?"
    ]
    
    print("=" * 100)
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 QUERY {i}: {query}\n")
        
        for chunk in agent_executor.stream(
            {"messages": [("user", query)]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                chunk["messages"][-1].pretty_print()
        
        print("=" * 100)