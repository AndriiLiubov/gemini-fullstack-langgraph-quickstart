import argparse
from pathlib import Path
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory to search for markdown files (default: current directory)",
    )
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--reasoning-model",
        default="llama-3.3-70b-versatile",
        help="Model for the final answer",
    )
    args = parser.parse_args()
    
    # Проверяем существование директории
    search_dir = Path(args.dir)
    if not search_dir.exists():
        raise ValueError(f"Directory {args.dir} does not exist")
    
    print(f"Searching in directory: {search_dir.absolute()}")

    state = {
        "messages": [HumanMessage(content=args.question)],
        "search_dir": str(search_dir.absolute()),
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(messages[-1].content)
        
        # Выводим источники
        sources = result.get("sources_gathered", [])
        if sources:
            print("\n" + "="*80)
            print("SOURCES USED:")
            print("="*80)
            unique_sources = {}
            for source in sources:
                if source["value"] not in unique_sources:
                    unique_sources[source["value"]] = source["label"]
            
            for i, (url, title) in enumerate(unique_sources.items(), 1):
                print(f"{i}. {title}")
                print(f"   {url}")


if __name__ == "__main__":
    main()