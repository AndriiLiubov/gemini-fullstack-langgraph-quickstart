from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from pathlib import Path


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Extracts the research topic from a list of conversation messages.
    
    Combines multiple messages into a single research topic string if there's 
    conversation history, otherwise returns the content of the single message.
    
    Args:
        messages: List of conversation messages (HumanMessage or AIMessage)
        
    Returns:
        A string containing the research topic or conversation history
    """
    # If only one message, use its content directly
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        # Combine multiple messages with speaker labels
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Creates a mapping from original URLs to simplified identifiers for citations.
    
    For local files, extracts just the filename. For web URLs, uses simple numeric
    identifiers. This ensures citations are readable and don't contain overly long
    or complex URLs.
    
    Args:
        urls_to_resolve: List of URL objects with 'web.uri' attribute
        id: Unique identifier for this search operation
        
    Returns:
        Dictionary mapping original URLs to simplified citation identifiers
    """
    if not urls_to_resolve:
        return {}
    
    urls = []
    for item in urls_to_resolve:
        if hasattr(item, 'web') and hasattr(item.web, 'uri'):
            urls.append(item.web.uri)
    
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            if url.startswith('file://'):
                path = url[7:] 
                filename = Path(path).name
                resolved_map[url] = filename
            else:
                resolved_map[url] = f"[{idx + 1}]"
    
    return resolved_map


def get_citations(response: Any, resolved_urls_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Extracts citation information from search response objects.
    
    Handles both real Gemini API responses and mock responses from local search.
    Creates citation objects with position indices and file references for
    proper citation insertion into the text.
    
    Args:
        response: Search response object (can be Gemini response or MockResponse)
        resolved_urls_map: Dictionary mapping URLs to simplified identifiers
        
    Returns:
        List of citation dictionaries, each containing:
        - start_index: Character position where citation starts
        - end_index: Character position where citation ends
        - segments: List of file/URL references for this citation
    """
    citations = []

    # Handle empty or invalid responses
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return citations

    candidate = response.candidates[0]
    
    # Check for grounding metadata structure
    if not hasattr(candidate, 'grounding_metadata') or not candidate.grounding_metadata:
        return citations
    
    # Get grounding chunks (files that were found)
    grounding_chunks = getattr(candidate.grounding_metadata, 'grounding_chunks', [])
    
    # Check if this is a mock response without proper grounding_supports
    has_grounding_supports = (
        hasattr(candidate.grounding_metadata, 'grounding_supports') and 
        candidate.grounding_metadata.grounding_supports
    )
    
    if not has_grounding_supports:
        # MockResponse case: Create simple citations for each file
        for i, chunk in enumerate(grounding_chunks):
            try:
                if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                    uri = chunk.web.uri
                    resolved_url = resolved_urls_map.get(uri, None)
                    
                    # Create citation for this file
                    citation = {
                        "start_index": i * 100,  # Placeholder positions
                        "end_index": (i + 1) * 100,
                        "segments": [{
                            "label": Path(uri[7:]).name if uri.startswith('file://') else "Source",
                            "short_url": resolved_url,
                            "value": uri,
                        }]
                    }
                    citations.append(citation)
            except Exception:
                continue
        return citations
    
    # Real response case: Process actual grounding supports
    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        # Skip if segment information is missing
        if not hasattr(support, "segment") or support.segment is None:
            continue

        # Extract position indices
        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        if support.segment.end_index is None:
            continue  # Skip if no end index

        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index
        citation["segments"] = []

        # Process each grounding chunk referenced by this support
        if hasattr(support, "grounding_chunk_indices") and support.grounding_chunk_indices:
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    
                    # Determine label based on URL type
                    if chunk.web.uri.startswith('file://'):
                        label = Path(chunk.web.uri[7:]).name
                    else:
                        label = (
                            chunk.web.title.split(".")[:-1][0] 
                            if chunk.web.title else "Source"
                        )
                    
                    citation["segments"].append({
                        "label": label,
                        "short_url": resolved_url,
                        "value": chunk.web.uri,
                    })
                except (IndexError, AttributeError):
                    # Skip invalid chunk references
                    continue
        
        if citation["segments"]:  # Only add if we have segments
            citations.append(citation)
    
    return citations


def insert_citation_markers(text: str, citations_list: List[Dict[str, Any]]) -> str:
    """
    Inserts citation markers into text at specified positions.
    
    Citations are inserted in reverse order (from end to beginning) to preserve
    correct character indices during insertion. Each citation appears as a
    simplified file reference in square brackets.
    
    Args:
        text: Original text to insert citations into
        citations_list: List of citation dictionaries with position indices
        
    Returns:
        Text with citation markers inserted at the appropriate positions
    """
    # Sort citations by end_index in descending order for safe insertion
    sorted_citations = sorted(
        citations_list, 
        key=lambda c: (c["end_index"], c["start_index"]), 
        reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        
        for segment in citation_info["segments"]:
            short_url = segment['short_url']
            if '/' in short_url:
                filename = short_url.split('/')[-1]
                marker_to_insert += f" `{filename}`"
            else:
                marker_to_insert += f" `{short_url}`"
        
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text