import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib


class LocalFileSearch:
    """Simple search engine for local markdown files without external dependencies."""
    
    def __init__(self, search_dir: str):
        self.search_dir = Path(search_dir)
        if not self.search_dir.exists():
            raise ValueError(f"Directory {search_dir} does not exist")
        self.file_index = {}
        self.content_index = {}
        
#        print(f"DEBUG: Initializing LocalFileSearch with directory: {self.search_dir}")
#        print(f"DEBUG: Directory exists: {self.search_dir.exists()}")
#        print(f"DEBUG: Directory is absolute: {self.search_dir.is_absolute()}")
#        print(f"DEBUG: Directory contents:")
        try:
            for item in self.search_dir.iterdir():
                print(f"  - {item.name}")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        self._index_files()
    
    def _parse_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse markdown file into sections without external libraries."""
        sections = []
        lines = content.split('\n')
        
        current_section = {
            "heading": "",
            "content": "",
            "line_numbers": [],
            "in_code_block": False
        }
        
        for i, line in enumerate(lines, 1):
            # Track code blocks
            stripped_line = line.strip()
            if stripped_line.startswith('```'):
                current_section["in_code_block"] = not current_section["in_code_block"]
                continue
            
            # Detect headings (simple detection)
            if (not current_section["in_code_block"] and 
                stripped_line.startswith('#') and 
                len(stripped_line) > 1 and 
                stripped_line[1] != '#'):  # Start of heading
                
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Extract heading level and text
                heading_match = re.match(r'^(#+)\s+(.+)$', stripped_line)
                if heading_match:
                    heading_level = len(heading_match.group(1))
                    heading_text = heading_match.group(2)
                    
                    current_section = {
                        "heading": heading_text,
                        "content": line + "\n",
                        "line_numbers": [i],
                        "in_code_block": False
                    }
                    continue
            
            # Add line to current section
            current_section["content"] += line + "\n"
            current_section["line_numbers"].append(i)
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _extract_plain_text(self, markdown_text: str) -> str:
        """Extract plain text from markdown (simplified)."""
        # Remove code blocks
        text = re.sub(r'```.*?```', '', markdown_text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`[^`]*`', '', text)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        # Remove bold/italic markers
        text = re.sub(r'[*_]{1,3}', '', text)
        # Remove headers markers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove blockquotes
        text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _index_files(self):
        """Index all markdown files in the directory."""
        md_files = list(self.search_dir.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files in {self.search_dir}")
        
        for file_path in md_files:
            try:
                # Generate unique ID for the file
                file_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse sections
                sections = self._parse_markdown_sections(content)
                
                # Extract plain text for search
                plain_text = self._extract_plain_text(content)
                
                # Extract headings
                headings = []
                for section in sections:
                    if section["heading"]:
                        headings.append({
                            "level": section["heading"].count('#') if section["heading"].startswith('#') else 1,
                            "text": section["heading"],
                            "line": section["line_numbers"][0] if section["line_numbers"] else 0
                        })
                
                self.file_index[file_id] = {
                    "path": str(file_path),
                    "relative_path": str(file_path.relative_to(self.search_dir)),
                    "headings": headings,
                    "sections": sections,
                    "full_content": content,
                    "plain_text": plain_text
                }
                
                # Index content by keywords (simple tokenization)
                words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', plain_text.lower())  # Only words with 3+ chars
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                for word, freq in word_freq.items():
                    if word not in self.content_index:
                        self.content_index[word] = []
                    self.content_index[word].append({
                        "file_id": file_id,
                        "count": freq
                    })
                    
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for query in indexed files."""
        # Extract meaningful query terms (3+ characters)
        query_terms = [term.lower() for term in re.findall(r'\b\w{3,}\b', query.lower())]
        
        if not query_terms:
            return []
        
        results = []
        file_scores = {}
        
        # Score files based on query term matches
        for term in query_terms:
            if term in self.content_index:
                for entry in self.content_index[term]:
                    file_id = entry["file_id"]
                    if file_id not in file_scores:
                        file_scores[file_id] = 0
                    file_scores[file_id] += entry["count"] * 2  # Weight exact matches
        
        # Also check for partial matches and context
        for file_id, file_info in self.file_index.items():
            content_lower = file_info["plain_text"].lower()
            
            # Score based on query presence in text
            for term in query_terms:
                if term in content_lower:
                    if file_id not in file_scores:
                        file_scores[file_id] = 0
                    file_scores[file_id] += content_lower.count(term)
            
            # Bonus score if all query terms appear in the file
            if all(term in content_lower for term in query_terms):
                file_scores[file_id] = file_scores.get(file_id, 0) + len(query_terms) * 5
        
        # Sort files by score
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        for file_id, score in sorted_files[:limit]:
            if score == 0:
                continue
                
            file_info = self.file_index[file_id]
            
            # Find relevant sections
            relevant_sections = []
            for section in file_info["sections"]:
                section_text = section["content"].lower()
                term_matches = sum(1 for term in query_terms if term in section_text)
                if term_matches > 0:
                    # Extract a snippet around the first match
                    snippet = section["content"]
                    if len(snippet) > 500:
                        # Try to find a good truncation point
                        for term in query_terms:
                            pos = snippet.lower().find(term)
                            if pos > 0:
                                start = max(0, pos - 100)
                                end = min(len(snippet), pos + 400)
                                snippet = "..." + snippet[start:end] + "..."
                                break
                        else:
                            snippet = snippet[:500] + "..."
                    
                    relevant_sections.append({
                        "heading": section["heading"],
                        "content": snippet,
                        "score": term_matches,
                        "line_numbers": section["line_numbers"]
                    })
            
            # Sort sections by relevance
            relevant_sections.sort(key=lambda x: x["score"], reverse=True)
            
            results.append({
                "title": file_info["relative_path"],
                "uri": f"file://{file_info['path']}",
                "file_id": file_id,
                "score": score,
                "snippets": relevant_sections[:3],  # Top 3 most relevant sections
                "full_content": file_info["full_content"]
            })
        
        return results
    
    def get_file_content(self, file_id: str) -> str:
        """Get full content of a file by its ID."""
        return self.file_index.get(file_id, {}).get("full_content", "")


# Mock classes remain the same
class MockWebInfo:
    def __init__(self, title: str, uri: str):
        self.title = title
        self.uri = uri


class MockGroundingChunk:
    def __init__(self, title: str, uri: str):
        self.web = MockWebInfo(title, uri)


class MockCandidate:
    def __init__(self, grounding_chunks):
        self.grounding_metadata = type('obj', (object,), {
            'grounding_chunks': grounding_chunks,
            'grounding_supports': []
        })()


class MockResponse:
    def __init__(self, grounding_chunks):
        self.candidates = [MockCandidate(grounding_chunks)]