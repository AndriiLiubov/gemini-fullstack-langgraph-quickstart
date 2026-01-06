import os
import re
from pathlib import Path

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_groq import ChatGroq
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.local_search import LocalFileSearch, MockGroundingChunk, MockResponse

load_dotenv()

# Check for required environment variables
if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set (required for LLM reasoning)")

# Global variable for local search engine instance
local_searcher = None


def get_local_searcher(search_dir: str) -> LocalFileSearch:
    """Get or create a local search engine instance.
    
    Args:
        search_dir: Directory path to search for markdown files
        
    Returns:
        LocalFileSearch instance initialized with the specified directory
    """
    global local_searcher
    if local_searcher is None or local_searcher.search_dir != Path(search_dir):
        local_searcher = LocalFileSearch(search_dir)
    return local_searcher


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generates search queries using Groq LLM for local file search.

    All reasoning is performed via Groq. The queries are optimized for searching
    in local markdown documentation files.

    Args:
        state: Current graph state containing the User's question and search directory
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # Check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Initialize Groq LLM
    llm = ChatGroq(
        model=configurable.llm_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt with enhanced instructions for local search
    current_date = get_current_date()
    search_dir = state.get("search_dir", "current directory")
    
    # Enhance instructions for local file search
    enhanced_instructions = query_writer_instructions + f"""
    
    IMPORTANT CONTEXT: You are searching in LOCAL MARKDOWN FILES located in directory: {search_dir}
    
    When generating search queries:
    1. Focus on keywords, concepts, and terminology likely to appear in technical documentation
    2. Consider file names, section headings, and technical terms
    3. Generate queries suitable for searching through markdown content
    4. Prioritize specific technical terms over general phrases
    """
    
    formatted_prompt = enhanced_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)

#    print(f"DEBUG generate_query: Received search_dir = {state.get('search_dir')}")
#   print(f"DEBUG generate_query: Will return search_dir = {state.get('search_dir', '.')}")
    
    
    return {
        "search_query": result.query,
        "search_dir": state.get("search_dir", ".") 
    }


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the research node.

    This is used to spawn multiple research nodes, one for each search query.
    The naming 'web_research' is kept for compatibility but now refers to local file research.

    Args:
        state: State containing generated search queries
        
    Returns:
        List of Send operations to the web_research node
    """
    search_dir = state.get("search_dir", ".")

#    print(f"DEBUG continue_to_web_research: Passing search_dir = {search_dir}")
    
    return [
        Send("web_research", {
            "search_query": search_query, 
            "id": int(idx),
            "search_dir": search_dir
        })
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Searches for information in local markdown files.

    This function replaces the original web search functionality with local file search.
    It indexes and searches markdown files in the specified directory, extracts relevant
    information, and formats it for further processing.

    Args:
        state: Current graph state containing the search query and ID
        config: Runnable configuration (not used here but required by interface)

    Returns:
        Dictionary with state updates including:
            - "sources_gathered": list of sources found in local files
            - "search_query": the original search queries
            - "web_research_result": list of retrieved and annotated text from local files
    """
    configurable = Configuration.from_runnable_config(config)
    search_dir = state.get("search_dir", ".")

#    print(f"=== DEBUG web_research ===")
#    print(f"Received search_dir: '{search_dir}'")
#    print(f"Query: '{state['search_query']}'")
#    print(f"ID: {state['id']}")
#    print(f"========================")
    
    # Initialize local search engine
    searcher = get_local_searcher(search_dir)
    
    # Execute search in local markdown files
#    print(f"DEBUG: Searching for query: '{state['search_query']}'")
    search_results = searcher.search(state["search_query"], limit=5)
    
#    print(f"DEBUG: Found {len(search_results)} search results")
    for result in search_results:
        print(f"  - {result['title']}")

    
    # Create mock grounding chunks for compatibility with existing citation logic
    grounding_chunks = []
    for result in search_results:
        chunk = MockGroundingChunk(
            title=result["title"],
            uri=result["uri"]
        )
        grounding_chunks.append(chunk)
    
    # Create mock response to maintain interface compatibility
    response = MockResponse(grounding_chunks)
    
    # Generate unique IDs for URLs (now file paths)
    resolved_urls = resolve_urls(grounding_chunks, state["id"])
    
    # Collect evidence from search results
    evidence_sources = []
    for result in search_results:
        # Format evidence text with file information
        evidence_text = f"File: {result['title']}\n"
        if result["snippets"]:
            for snippet in result["snippets"][:2]:  # Take up to 2 most relevant snippets
                if snippet["heading"]:
                    evidence_text += f"\n## {snippet['heading']}\n"
                evidence_text += snippet["content"][:300] + "...\n"
        evidence_sources.append(evidence_text)
    
    # Combine all evidence sources
    raw_evidence = "\n\n---\n\n".join(evidence_sources)
    
    # Summarize evidence using Groq LLM
    llm = ChatGroq(
        model=configurable.llm_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    summary_prompt = f"""
    Analyze and summarize the following information extracted from local markdown files.
    
    Search Query: "{state['search_query']}"
    
    Information from files:
    {raw_evidence}
    
    Provide a concise, factual summary that:
    1. Extracts the most relevant information related to the search query
    2. Preserves technical accuracy
    3. Identifies key concepts and their explanations
    4. Notes any limitations or gaps in the found information
    """
    
    summary = llm.invoke(summary_prompt).content
    
    # Attach citations to the summary
    citations = get_citations(response, resolved_urls)
    final_text = insert_citation_markers(summary, citations)
    sources_gathered = [item for c in citations for item in c["segments"]]
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [final_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract the follow-up query
    in JSON format. Enhanced for local file search context.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including:
            - is_sufficient: whether current information is sufficient
            - knowledge_gap: description of missing information
            - follow_up_queries: list of generated follow-up queries
            - research_loop_count: current loop iteration
            - number_of_ran_queries: total queries executed so far
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.llm_model)
    
    # Enhance reflection instructions for local search context
    enhanced_reflection = reflection_instructions + """
    
    LOCAL SEARCH CONTEXT: You are researching in local markdown files. Consider:
    1. What specific files or document sections might contain the missing information?
    2. What terminology, function names, or technical concepts should be searched for?
    3. Are there related files or adjacent documentation that might have been missed?
    4. Consider searching for specific code examples, API references, or configuration snippets
    5. Think about alternative terminology or synonyms used in the documentation
    """
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = enhanced_reflection.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # Initialize reasoning model
    llm = ChatGroq(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
        "search_dir": state.get("search_dir", "."),  
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops
    and the sufficiency of current information.

    Args:
        state: Current graph state containing the research loop count and sufficiency flag
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
        or list of Send operations for parallel follow-up queries
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get maximum research loops from state or configuration
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    
    # Get search_dir from the state
    search_dir = state.get("search_dir", ".")
    
#    print(f"DEBUG evaluate_research: search_dir = {search_dir}")
    
    # Determine next step based on sufficiency and loop count
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # Generate follow-up research queries in parallel
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                    "search_dir": search_dir, 
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research answer.

    Prepares the final output by combining all gathered information into a
    well-structured research report with proper citations to local files.

    Args:
        state: Current graph state containing all research results and sources
        config: Configuration for the runnable, including LLM model settings

    Returns:
        Dictionary with state update, including:
            - messages: final answer message
            - sources_gathered: deduplicated list of sources used
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model", configurable.llm_model)
    
    # Enhance answer instructions for local file context
    enhanced_answer = answer_instructions + """
    
    LOCAL FILE CONTEXT: Your research was conducted in local markdown files. 
    
    CRITICAL FORMATTING RULES FOR FILE CITATIONS:
    1. When citing files, use ONLY the filename without the full path
    2. Example: Use `add-human-in-the-loop.md` NOT `how-tos/human_in_the_loop/add-human-in-the-loop.md`
    3. Example: Use `functional_api.md` NOT `concepts/functional_api.md`
    4. Example: Use `graph-api.md` NOT `how-tos/graph-api.md`
    5. NEVER include `file://` or full directory paths in citations
    6. NEVER repeat the same file path multiple times
    7. Use simple backticks for file names: `filename.md`
    
    Examples of CORRECT citations:
    - As mentioned in `add-human-in-the-loop.md`...
    - According to `functional_api.md`...
    - The `graph-api.md` file explains...
    
    Examples of WRONG citations:
    - As mentioned in `how-tos/human_in_the_loop/add-human-in-the-loop.md`...
    - According to `file:///Users/.../functional_api.md`...
    - The `how-tos/graph-api.md` file explains...
    
    Before writing your final answer, check ALL file citations and:
    1. Remove any `file://` prefixes
    2. Remove directory paths (keep only filename)
    3. Remove duplicate file mentions
    4. Format as `filename.md`
    """
    
    # Format the final answer prompt
    current_date = get_current_date()
    
    summaries = state.get("web_research_result", [])
    cleaned_summaries = []
    
    for summary in summaries:
        cleaned = summary
        cleaned = re.sub(r'file://[^`\s]+/', '', cleaned)
        cleaned = re.sub(r'([^/\s]+/){2,}', '', cleaned)
        cleaned_summaries.append(cleaned)
    
    formatted_prompt = enhanced_answer.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(cleaned_summaries),
    )
    
    # Initialize reasoning model
    llm = ChatGroq(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)
    
    # Post-process: clean up any remaining file paths in the answer
    final_content = result.content
    
    file_pattern = r'`([^`]+/)*([^/`]+\.md)`'
    
    def replace_file_path(match):
        filename = match.group(2)
        return f'`{filename}`'
    
    final_content = re.sub(file_pattern, replace_file_path, final_content)
    
    path_pattern = r'([^/\s]+/)+([^/\s]+\.md)'
    final_content = re.sub(path_pattern, r'\2', final_content)
    
    final_content = final_content.replace('file://', '')
    
    for source in state["sources_gathered"]:
        if source["short_url"] in final_content:
            # Используем только имя файла
            filename = Path(source["value"]).name if source["value"].startswith('file://') else source["short_url"]
            final_content = final_content.replace(source["short_url"], filename)
    
    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": state["sources_gathered"],
    }


# Create the Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes in the research workflow
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Define the graph edges and flow control

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")

# Add conditional edge to continue with search queries in parallel branches
builder.add_conditional_edges(
    "generate_query", 
    continue_to_web_research, 
    ["web_research"]
)

# After web research, reflect on the gathered information
builder.add_edge("web_research", "reflection")

# Evaluate the research and decide next step
builder.add_conditional_edges(
    "reflection", 
    evaluate_research, 
    ["web_research", "finalize_answer"]
)

# Finalize the answer and end the graph
builder.add_edge("finalize_answer", END)

# Compile the graph into an executable agent
graph = builder.compile(name="local-file-research-agent")