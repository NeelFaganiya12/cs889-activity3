import streamlit as st
import json
import os
import random
import requests
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict
import re

# -----------------------------
# Load environment variables
# -----------------------------
from pathlib import Path
load_dotenv(dotenv_path=Path(".env"))

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]

GEMINI_API_KEY = random.choice([k for k in GEMINI_KEYS if k])
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# OpenAlex API functions
# -----------------------------
def search_openalex(query: str, limit: int = 20, year_filter: str = None, min_citations: int = None):
    """
    Search OpenAlex API for papers using the works endpoint
    Documentation: https://docs.openalex.org/api-entities/works/search-works
    OpenAlex has generous rate limits (10 requests/second) - no rate limit handling needed
    """
    query = query.strip()
    if not query:
        return []
    
    # Clean query - remove redundant words (but keep the query meaningful)
    query = query.replace(" papers", "").replace(" paper", "")
    query = query.replace(" articles", "").replace(" article", "")
    query = query.strip()
    
    # Ensure query is still valid after cleaning
    if not query or len(query) < 2:
        st.warning("⚠️ Search query is too short. Please enter at least 2 characters.")
        return []
    
    # Ensure limit is within valid range (max 200 per API docs, but we'll use 25 for performance)
    limit = max(1, min(limit, 25))
    
    # Use the OpenAlex works endpoint
    url = "https://api.openalex.org/works"
    
    # Build parameters according to OpenAlex API documentation
    # OpenAlex uses 'search' parameter for full-text search
    params = {
        "search": query,
        "per_page": limit
    }
    
    # Note: OpenAlex doesn't support 'select' parameter in the same way
    # We'll get all fields and extract what we need
    
    # Add optional filters
    filters = []
    
    if year_filter:
        # Parse year filter (format: "2016-2020" or "2019")
        if "-" in year_filter:
            years = year_filter.split("-")
            try:
                start_year = int(years[0])
                end_year = int(years[1])
                filters.append(f"publication_year:{start_year}-{end_year}")
            except:
                pass
        else:
            try:
                year = int(year_filter)
                filters.append(f"publication_year:{year}")
            except:
                pass
    
    if min_citations:
        filters.append(f"cited_by_count:>={min_citations}")
    
    # Combine filters with comma separation
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        # Make request with proper headers
        headers = {
            'User-Agent': 'LitSense/1.0 (mailto:user@example.com)'  # OpenAlex prefers identifying user agents
        }
        
        # Debug: Log the request URL (remove in production)
        # st.write(f"Debug: Requesting {url} with params: {params}")
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # OpenAlex returns: meta, results[]
            papers_data = data.get('results', [])
            
            # If no results, return empty list
            if not papers_data:
                return []
            
            papers = []
            for paper in papers_data:
                # Extract title
                title = paper.get('display_name') or paper.get('title', 'Untitled')
                
                # Extract authors from authorships array
                authors = []
                if paper.get('authorships'):
                    for authorship in paper.get('authorships', []):
                        author = authorship.get('author')
                        if author:
                            author_name = author.get('display_name', '')
                            if author_name:
                                authors.append(author_name)
                
                # Extract year
                year = paper.get('publication_year') or 0
                
                # Extract abstract
                abstract = paper.get('abstract', 'No abstract available')
                # OpenAlex abstracts are sometimes in inverted format, check if it starts with inverted
                if abstract.startswith("InvertedAbstract"):
                    # Try to extract the actual abstract
                    abstract = abstract.replace("InvertedAbstract", "").strip()
                
                # Extract journal/venue from primary_location
                journal = "N/A"
                if paper.get('primary_location'):
                    source = paper.get('primary_location', {}).get('source')
                    if source:
                        journal = source.get('display_name', 'N/A')
                
                # Extract DOI
                doi = paper.get('doi', '').replace('https://doi.org/', '') if paper.get('doi') else ''
                
                # Extract concepts (similar to fields of study)
                concepts = []
                if paper.get('concepts'):
                    concepts = [concept.get('display_name', '') for concept in paper.get('concepts', [])[:5] if concept.get('display_name')]
                
                # Extract citation count
                citation_count = paper.get('cited_by_count', 0)
                
                # Extract OpenAlex ID (URL format: https://openalex.org/W123456789)
                openalex_id = paper.get('id', '').replace('https://openalex.org/', '') if paper.get('id') else ''
                
                # Extract URL
                url_link = paper.get('primary_location', {}).get('landing_page_url', '') or paper.get('id', '')
                
                # Use OpenAlex ID as primary identifier, or create unique hash
                unique_id = openalex_id or hash(f"{title}_{year}")
                
                paper_obj = {
                    "id": unique_id,
                    "paperId": openalex_id,  # Store OpenAlex ID
                    "title": title,
                    "authors": authors if authors else ["Unknown"],
                    "year": year,
                    "abstract": abstract if abstract else "No abstract available",
                    "journal": journal,
                    "doi": doi,
                    "keywords": concepts,  # Use concepts as keywords
                    "citation_count": citation_count,
                    "url": url_link,
                    "fieldsOfStudy": concepts,
                    "publicationTypes": []
                }
                papers.append(paper_obj)
            
            return papers
        
        elif response.status_code == 429:
            # OpenAlex rate limit (very rare, but handle it)
            st.warning("⏰ OpenAlex rate limit reached. Please wait a moment and try again.")
            return []
        
        elif response.status_code == 400:
            # Try to get more details from the error response
            try:
                error_data = response.json()
                error_msg = error_data.get('message', 'Invalid search query')
                st.error(f"❌ OpenAlex API error: {error_msg}")
                st.info(f"💡 **Troubleshooting:**\n- Make sure your search query is not empty\n- Try simpler search terms\n- Check if special characters are causing issues")
            except:
                # If we can't parse JSON, show the raw response text
                try:
                    error_text = response.text[:200]  # First 200 chars
                    st.error(f"❌ Invalid search query. API response: {error_text}")
                except:
                    st.error("❌ Invalid search query. Try a different search term.")
            return []
        
        elif response.status_code == 429:
            # OpenAlex rate limit (very rare, but handle it)
            st.warning("⏰ OpenAlex rate limit reached. Please wait a moment and try again.")
            return []
        
        else:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f'API error: {response.status_code}')
                st.error(f"❌ OpenAlex API error ({response.status_code}): {error_msg}")
            except:
                st.error(f"❌ OpenAlex API error: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error connecting to OpenAlex: {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return []

# -----------------------------
# AI Clustering function
# -----------------------------
def cluster_papers(papers, search_query):
    """Use AI to cluster papers into thematic groups"""
    if not papers or not GEMINI_API_KEY:
        return {}
    
    try:
        # Create summary of papers
        papers_summary = "\n\n".join([
            f"Paper {i+1}: {p['title']} - {p['abstract'][:150]}..."
            for i, p in enumerate(papers[:15])
        ])
        
        prompt = f"""Given these research papers about "{search_query}", organize them into 3-5 thematic clusters.

Papers:
{papers_summary}

For each cluster, provide:
1. A short cluster name (2-4 words)
2. The paper numbers that belong to it
3. Key topics/keywords for that cluster

Format:
CLUSTER 1: [name]
Papers: [numbers]
Topics: [keywords]

CLUSTER 2: [name]
Papers: [numbers]
Topics: [keywords]
..."""
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse clusters
        clusters = {}
        current_cluster = None
        
        for line in text.split('\n'):
            if 'CLUSTER' in line.upper() or 'CLUSTER' in line:
                # Extract cluster name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_cluster = parts[1].strip()
                    clusters[current_cluster] = {"papers": [], "topics": []}
            elif current_cluster and 'Papers:' in line:
                # Extract paper numbers
                numbers = re.findall(r'\d+', line)
                clusters[current_cluster]["papers"] = [int(n) - 1 for n in numbers if int(n) <= len(papers)]
            elif current_cluster and 'Topics:' in line:
                # Extract topics
                topics = line.split('Topics:')[1].strip()
                clusters[current_cluster]["topics"] = [t.strip() for t in topics.split(',')[:5]]
        
        return clusters
    except Exception as e:
        return {}

# -----------------------------
# AI helper - Explain relevance
# -----------------------------
def explain_relevance(paper, user_query=""):
    """Use Gemini to explain paper relevance"""
    authors_str = ", ".join(paper.get('authors', ['Unknown']))
    paper_info_query = f"Explain the relevance of this paper: {paper.get('title', 'N/A')} by {authors_str} ({paper.get('year', 'N/A')})"
    
    if user_query and user_query.strip():
        full_query = f"{paper_info_query}. User's research interest: {user_query}"
    else:
        full_query = paper_info_query
    
    prompt = f"""How is this paper relevant here?

Paper title: {paper.get('title', 'N/A')}
Authors: {', '.join(paper.get('authors', ['Unknown']))}
Year: {paper.get('year', 'N/A')}
Journal: {paper.get('journal', 'N/A')}

Abstract:
{paper.get('abstract', 'No abstract available')}

User's search topic: {user_query if user_query else 'General research'}

Please explain in 3-4 sentences how this paper is relevant to the user's search topic.
Focus on conceptual relevance and what this paper contributes to the research area.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# AI Relevance Ranking
# -----------------------------
def rank_papers_by_relevance(papers, search_query):
    """Rank papers by relevance to search query"""
    if not papers or not GEMINI_API_KEY:
        return papers
    
    try:
        papers_list = "\n".join([
            f"{i+1}. {p['title']}"
            for i, p in enumerate(papers[:10])
        ])
        
        prompt = f"""Rank these papers by relevance to "{search_query}" (most relevant first).

Papers:
{papers_list}

Return only the numbers in order of relevance, separated by commas."""
        
        response = model.generate_content(prompt)
        ranked_indices = [int(x.strip()) - 1 for x in response.text.split(',') if x.strip().isdigit()]
        
        # Reorder papers
        ranked = [papers[i] for i in ranked_indices if 0 <= i < len(papers)]
        # Add any papers not in ranking (avoid duplicates)
        ranked_indices_set = set(ranked_indices)
        remaining = [p for i, p in enumerate(papers) if i not in ranked_indices_set]
        return ranked + remaining
    except:
        return papers

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LitSense", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {}
if "cached_papers" not in st.session_state:
    st.session_state.cached_papers = {}
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""
if "rate_limit_time" not in st.session_state:
    st.session_state.rate_limit_time = None
if "selected_paper_id" not in st.session_state:
    st.session_state.selected_paper_id = None
if "paper_feedback" not in st.session_state:
    st.session_state.paper_feedback = {}
if "clusters" not in st.session_state:
    st.session_state.clusters = {}
if "ranked_papers" not in st.session_state:
    st.session_state.ranked_papers = []
if "last_data_source" not in st.session_state:
    st.session_state.last_data_source = None
if "all_loaded_papers" not in st.session_state:
    st.session_state.all_loaded_papers = []  # Store all papers that have been loaded for feedback tracking
if "current_papers" not in st.session_state:
    st.session_state.current_papers = []  # Store current papers for paper details access

# Header
col_header1, col_header2 = st.columns([4, 1])
with col_header1:
    st.title("📚 Literature Review Assistant")
    st.caption("Search → Explore → Review → Refine")
with col_header2:
    if st.session_state.selected_papers:
        st.metric("Saved", len(st.session_state.selected_papers))

# Main layout: 3 columns
col_search, col_results, col_details = st.columns([1, 2, 1.5])

# ==================== LEFT COLUMN: SEARCH & STEERING ====================
with col_search:
    st.header("🔍 Search & Filter")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Search Online", "Local Papers"],
        help="Choose between live online search or local papers",
        key="data_source_selector"
    )
    
    # Note about loading time
    if data_source == "Search Online":
        st.caption("⏱️ Note: Processing may take 10-15 seconds (AI clustering & ranking)")
    else:
        st.caption("⏱️ Note: Processing may take 10-15 seconds (AI clustering & ranking)")
    
    # Clear selected paper when switching data sources
    if data_source != st.session_state.get('last_data_source'):
        if st.session_state.get('last_data_source') is not None:
            st.session_state.selected_paper_id = None
            # Clear AI explanations when switching to avoid showing wrong explanations
            st.session_state.ai_explanations = {}
            # Clear rate limit timer when switching
            st.session_state.rate_limit_time = None
            # Clear current_papers when switching to ensure fresh data
            if 'current_papers' in st.session_state:
                del st.session_state.current_papers
    
    st.divider()
    
    if data_source == "Search Online":
        search_query = st.text_input(
            "Research Topic",
            placeholder="e.g. machine learning, transformer architectures",
            help="Enter your research topic to search online",
            key="main_search"
        )
        
        # Search button
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
    else:
        # Local papers mode
        col_filter1, col_filter2 = st.columns([3, 1])
        with col_filter1:
            search_query = st.text_input(
                "Filter Papers",
                placeholder="Search within local papers...",
                help="Filter papers by title, abstract, or keywords",
                key="local_search"
            )
        with col_filter2:
            # Show clear button only if filter is active
            if st.session_state.get('local_search', ''):
                if st.button("🗑️ Clear", use_container_width=True, help="Clear the filter"):
                    st.session_state.local_search = ""
                    st.rerun()
            else:
                st.write("")  # Empty space to maintain layout
        
        search_clicked = st.button("🔍 Filter", type="primary", use_container_width=True)
    
    # OpenAlex has generous rate limits (10 req/sec) - no rate limit warnings needed
    
    st.divider()
    
    # Year filter (only for Search Online)
    if data_source == "Search Online":
        year_filter_enabled = st.checkbox("Filter by Year", value=False)
        if year_filter_enabled:
            year_range = st.slider(
                "Publication Year",
                min_value=2000,
                max_value=2025,
                value=(2020, 2025)
            )
            st.session_state.year_filter = year_range
        else:
            if "year_filter" in st.session_state:
                del st.session_state.year_filter
    
    st.divider()
    
    # Cached searches (only for Search Online)
    if data_source == "Search Online" and st.session_state.cached_papers:
        st.subheader("📦 Recent Searches")
        for cached_query in list(st.session_state.cached_papers.keys())[:5]:
            if st.button(f"📄 {cached_query[:30]}...", key=f"cache_{cached_query}", use_container_width=True):
                st.session_state.last_search_query = cached_query
                st.rerun()
    
    # Show local papers info
    if data_source == "Local Papers":
        st.info("💡 **Local Papers Mode**\n\nShowing local papers. Use the filter box above to search within them.")

# ==================== MIDDLE COLUMN: RESULTS (CLUSTERS & QUEUE) ====================
with col_results:
    papers = []
    
    # Load papers based on data source
    if data_source == "Local Papers":
        # Load from local JSON file
        @st.cache_data
        def load_local_papers():
            try:
                with open("papers.json", "r") as f:
                    data = json.load(f)
                local_papers = data.get("references", [])
                
                # Normalize local papers to match expected format
                normalized = []
                for idx, p in enumerate(local_papers, 1):
                    normalized_paper = {
                        "id": p.get('id', idx),
                        "title": p.get('title', 'Untitled'),
                        "authors": p.get('authors', ['Unknown']),
                        "year": p.get('year', 0),
                        "abstract": p.get('abstract', 'No abstract available'),
                        "journal": p.get('journal', 'N/A'),
                        "doi": p.get('doi', ''),
                        "keywords": p.get('keywords', []),
                        "citation_count": 0,  # Local papers don't have citation data
                        "url": "",
                        "volume": p.get('volume', ''),
                        "issue": p.get('issue', ''),
                        "pages": p.get('pages', '')
                    }
                    normalized.append(normalized_paper)
                return normalized
            except FileNotFoundError:
                st.error("❌ Local papers file not found!")
                return []
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON in local papers file: {str(e)}")
                return []
        
        all_local_papers = load_local_papers()
        
        # Filter local papers if search query provided
        # Get the actual search query value from session state
        local_search_value = st.session_state.get('local_search', '')
        if local_search_value:
            query_lower = local_search_value.lower()
            papers = [
                p for p in all_local_papers
                if (query_lower in p.get('title', '').lower() or
                    query_lower in p.get('abstract', '').lower() or
                    any(query_lower in kw.lower() for kw in p.get('keywords', [])))
            ]
        else:
            papers = all_local_papers
        
        # Generate clusters and ranking for local papers (only if papers loaded)
        if papers:
            # Regenerate clusters/ranking when switching to local papers or on new search
            local_search_value = st.session_state.get('local_search', '')
            if data_source != st.session_state.get('last_data_source') or search_clicked or not st.session_state.clusters:
                with st.spinner("🤖 Organizing papers into clusters..."):
                    cluster_query = local_search_value if local_search_value else "research papers"
                    st.session_state.clusters = cluster_papers(papers, cluster_query)
                
                with st.spinner("📊 Ranking papers by relevance..."):
                    rank_query = local_search_value if local_search_value else "general research"
                    st.session_state.ranked_papers = rank_papers_by_relevance(papers, rank_query)
            
            st.session_state.last_data_source = data_source
    
    else:
        # Search Online mode
        # Get year filter if set
        year_filter = None
        if "year_filter" in st.session_state:
            year_range = st.session_state.year_filter
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                if year_range[0] != year_range[1]:
                    year_filter = f"{year_range[0]}-{year_range[1]}"
                else:
                    year_filter = str(year_range[0])
        
        # Perform search - OpenAlex has generous rate limits, no need for rate limit checks
        papers = []
        if search_query and (search_query != st.session_state.last_search_query or search_clicked):
            with st.spinner("🔍 Searching online..."):
                papers = search_openalex(search_query, limit=20, year_filter=year_filter)
                if papers:
                    st.session_state.cached_papers[search_query.lower()] = papers
                    st.session_state.last_search_query = search_query
                    
                    # Rank papers (clustering removed for online search due to issues)
                    with st.spinner("📊 Ranking papers by relevance..."):
                        ranked_result = rank_papers_by_relevance(papers, search_query)
                        # Remove duplicates from ranked results
                        seen_ids = set()
                        unique_ranked = []
                        for p in ranked_result:
                            pid = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
                            if pid not in seen_ids:
                                seen_ids.add(pid)
                                unique_ranked.append(p)
                        st.session_state.ranked_papers = unique_ranked
                    # Clear clusters for online search
                    st.session_state.clusters = {}
        elif search_query and search_query.lower() in st.session_state.cached_papers:
            papers = st.session_state.cached_papers[search_query.lower()]
            # If we have ranked papers for this query, use those instead
            if st.session_state.ranked_papers:
                # Check if ranked papers match the current query
                pass  # ranked_papers will be used in display
    
    # Note: Filters (include/exclude keywords, scope) have been removed per user request
    
    # Store papers in session state for feedback tracking and paper details access
    if papers:
        # Store current papers in session state so paper details can access them
        st.session_state.current_papers = papers
        
        # Update all_loaded_papers with current papers (avoid duplicates)
        # Use paperId or create unique hash for comparison
        seen_ids = set()
        for p in st.session_state.all_loaded_papers:
            pid = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
            seen_ids.add(pid)
        
        for p in papers:
            paper_id = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
            if paper_id not in seen_ids:
                st.session_state.all_loaded_papers.append(p)
                seen_ids.add(paper_id)
    else:
        # Clear current papers if no papers found
        if 'current_papers' in st.session_state:
            del st.session_state.current_papers
    
    # Display results
    # Ensure papers list doesn't have duplicates before displaying
    if papers:
        # Remove duplicates from papers list
        seen_paper_ids = set()
        unique_papers = []
        for p in papers:
            paper_id = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
            if paper_id not in seen_paper_ids:
                seen_paper_ids.add(paper_id)
                unique_papers.append(p)
        papers = unique_papers
        
        st.header(f"📄 {len(papers)} Papers Found")
        
        # For Local Papers: Show both Clusters and Review Queue
        # For Search Online: Show only Review Queue (clustering has issues)
        if data_source == "Local Papers":
            # Tabs for Clusters and Review Queue
            tab_clusters, tab_queue = st.tabs(["📚 Clusters", "📋 Review Queue"])
            
            with tab_clusters:
                if st.session_state.clusters:
                    for cluster_name, cluster_data in st.session_state.clusters.items():
                        cluster_papers_list = [papers[i] for i in cluster_data["papers"] if 0 <= i < len(papers)]
                        if cluster_papers_list:
                            with st.expander(f"**{cluster_name}** ({len(cluster_papers_list)} papers)", expanded=False):
                                if cluster_data.get("topics"):
                                    st.caption(f"Topics: {', '.join(cluster_data['topics'][:5])}")
                                
                                for cluster_idx, paper in enumerate(cluster_papers_list):
                                    # Get paper ID - use paperId if available, otherwise use id or hash
                                    paper_id = paper.get('paperId') or paper.get('id') or hash(f"{paper.get('title', '')}_{paper.get('year', '')}")
                                    # Create unique key using cluster name and index
                                    unique_cluster_key = f"select_{cluster_name}_{cluster_idx}_{paper_id}"
                                    col_paper1, col_paper2 = st.columns([4, 1])
                                    
                                    with col_paper1:
                                        if st.button(f"📄 {paper['title'][:60]}...", key=unique_cluster_key, use_container_width=True):
                                            st.session_state.selected_paper_id = paper_id
                                            st.rerun()
                                    
                                    with col_paper2:
                                        if paper.get('citation_count'):
                                            st.caption(f"⭐ {paper['citation_count']}")
                                    
                                    journal_display = paper.get('journal', 'N/A')
                                    if paper.get('volume') or paper.get('issue'):
                                        journal_display += f" ({paper.get('year', 'N/A')})"
                                    else:
                                        journal_display += f" • {paper.get('year', 'N/A')}"
                                    st.caption(journal_display)
                                    st.divider()
                else:
                    st.info("Clustering in progress...")
            
            with tab_queue:
                ranked = st.session_state.ranked_papers if st.session_state.ranked_papers else papers
                
                st.caption("Papers ranked by relevance to your search")
                
                for idx, paper in enumerate(ranked, 1):
                    # Get paper ID - use paperId if available, otherwise use id or hash
                    # This must match exactly how we calculate it in paper details section
                    paper_id = paper.get('paperId') or paper.get('id')
                    if paper_id is None:
                        paper_id = hash(f"{paper.get('title', '')}_{paper.get('year', '')}")
                    # Ensure paper_id is a consistent type (string for OpenAlex IDs, keep as-is for others)
                    if isinstance(paper_id, str) and paper_id.startswith('W'):
                        # OpenAlex ID - keep as string
                        pass
                    elif paper_id is None:
                        paper_id = hash(f"{paper.get('title', '')}_{paper.get('year', '')}")
                    # Create unique key using index to avoid duplicates
                    unique_key = f"view_queue_{idx}_{paper_id}"
                    
                    # Paper card
                    paper_selected = st.session_state.selected_paper_id == paper_id
                    card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                    
                    with st.container():
                        col_q1, col_q2 = st.columns([4, 1])
                        
                        with col_q1:
                            st.markdown(f"**{idx}. {paper['title']}**")
                            st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                            if paper.get('citation_count'):
                                st.caption(f"⭐ {paper['citation_count']} citations")
                            elif paper.get('keywords'):
                                st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                        
                        with col_q2:
                            if st.button("View", key=unique_key, use_container_width=True):
                                # Store the paper ID consistently
                                st.session_state.selected_paper_id = paper_id
                                st.rerun()
                        
                        st.divider()
        else:
            # Search Online: Only Review Queue (no clusters)
            # Use ranked papers if available, otherwise use papers
            # But ensure we don't show duplicates
            if st.session_state.ranked_papers:
                ranked = st.session_state.ranked_papers
            else:
                ranked = papers
            
            st.caption("Papers ranked by relevance to your search")
            
            # Remove duplicates based on paper ID
            seen_paper_ids = set()
            unique_ranked = []
            for paper in ranked:
                paper_id = paper.get('paperId') or paper.get('id') or hash(f"{paper.get('title', '')}_{paper.get('year', '')}")
                if paper_id not in seen_paper_ids:
                    seen_paper_ids.add(paper_id)
                    unique_ranked.append(paper)
            
            for idx, paper in enumerate(unique_ranked, 1):
                # Get paper ID - prioritize paperId for OpenAlex papers
                # This must match exactly how we calculate it in paper details section
                paper_id = None
                # For OpenAlex papers, use paperId first (it's the OpenAlex ID like "W123456789")
                if paper.get('paperId') and isinstance(paper.get('paperId'), str) and paper.get('paperId').startswith('W'):
                    paper_id = paper.get('paperId')
                else:
                    # For other papers, use paperId or id or hash
                    paper_id = paper.get('paperId') or paper.get('id')
                    if paper_id is None:
                        paper_id = hash(f"{paper.get('title', '')}_{paper.get('year', '')}")
                
                # Create unique key using index to avoid duplicates
                unique_key = f"view_scholar_{idx}_{paper_id}"
                
                # Paper card
                paper_selected = st.session_state.selected_paper_id == paper_id
                card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                
                with st.container():
                    col_q1, col_q2 = st.columns([4, 1])
                    
                    with col_q1:
                        st.markdown(f"**{idx}. {paper['title']}**")
                        st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                        if paper.get('citation_count'):
                            st.caption(f"⭐ {paper['citation_count']} citations")
                        elif paper.get('keywords'):
                            st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                    
                    with col_q2:
                        if st.button("View", key=unique_key, use_container_width=True):
                            # Store the paper ID consistently - use paperId for OpenAlex papers
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
                    
                    st.divider()
    else:
        if data_source == "Local Papers":
            st.info("📚 **Local Papers Mode**\n\nAll local papers are shown. Use the filter box to search within them.")
        else:
            st.info("👆 Enter a search query to find papers online")

# ==================== RIGHT COLUMN: SELECTED PAPER DETAILS ====================
with col_details:
    st.header("📖 Paper Details")
    
    # Find selected paper
    selected_paper = None
    all_papers = []
    
    # Get papers from current data source only
    # Only show paper details if the selected paper belongs to the current data source
    all_papers = []
    
    # Get papers based on current data source
    if data_source == "Local Papers":
        # For local papers, check current_papers first (most recent)
        if st.session_state.get('current_papers'):
            # Filter to only local papers (those without OpenAlex paperId starting with 'W')
            all_papers = [p for p in st.session_state.current_papers 
                         if not p.get('paperId') or not (isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W'))]
        elif st.session_state.get('all_loaded_papers'):
            # Filter to only local papers (those without OpenAlex paperId format)
            all_papers = [p for p in st.session_state.all_loaded_papers 
                         if not p.get('paperId') or not (isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W'))]
    else:
        # For Search Online, prioritize ranked_papers, then current_papers
        # Use ranked_papers first as they're most recent and definitely from Search Online
        if st.session_state.ranked_papers:
            # Use ranked papers (these are definitely from Search Online)
            all_papers = st.session_state.ranked_papers
        elif st.session_state.get('current_papers'):
            # Check if current_papers contains online papers (have paperId starting with 'W')
            online_papers = [p for p in st.session_state.current_papers 
                           if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
            if online_papers:
                all_papers = online_papers
            else:
                # If no online papers in current_papers, try all_loaded_papers
                if st.session_state.get('all_loaded_papers'):
                    all_papers = [p for p in st.session_state.all_loaded_papers 
                                 if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
                else:
                    all_papers = []
        else:
            # No papers available yet
            all_papers = []
    
    # Only show paper details if we have papers and a selected paper ID
    # AND the selected paper belongs to the current data source
    if st.session_state.selected_paper_id and all_papers:
        selected_id = st.session_state.selected_paper_id
        
        # Normalize selected_id for comparison
        selected_id_str = str(selected_id) if selected_id else None
        
        for p in all_papers:
            # Calculate ID the same way as when setting it (must match View button logic exactly)
            paper_id = None
            # For OpenAlex papers, prioritize paperId (it's the OpenAlex ID like "W123456789")
            if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W'):
                paper_id = p.get('paperId')
            else:
                # For other papers, use paperId or id or hash
                paper_id = p.get('paperId') or p.get('id')
                if paper_id is None:
                    paper_id = hash(f"{p.get('title', '')}_{p.get('year', '')}")
            
            paper_id_str = str(paper_id) if paper_id else None
            
            # Try multiple matching strategies (prioritize most specific first)
            # 1. Direct match (exact) - this should work for OpenAlex IDs
            if paper_id == selected_id:
                selected_paper = p
                break
            
            # 2. String comparison (handles type mismatches)
            if paper_id_str and selected_id_str and paper_id_str == selected_id_str:
                selected_paper = p
                break
            
            # 3. For OpenAlex papers (IDs starting with 'W'), match paperId directly
            if isinstance(selected_id, str) and selected_id.startswith('W'):
                paper_paperId = p.get('paperId')
                if paper_paperId == selected_id:
                    selected_paper = p
                    break
                if isinstance(paper_paperId, str) and paper_paperId == selected_id:
                    selected_paper = p
                    break
                # Also try string comparison
                if paper_paperId and str(paper_paperId) == selected_id:
                    selected_paper = p
                    break
            
            # 4. Match by hash of title+year as fallback
            title_hash = hash(f"{p.get('title', '')}_{p.get('year', '')}")
            if title_hash == selected_id or (selected_id_str and str(title_hash) == selected_id_str):
                selected_paper = p
                break
    
    if selected_paper:
        # Paper header
        st.markdown(f"### {selected_paper['title']}")
        
        # Authors
        authors_str = ', '.join(selected_paper.get('authors', ['Unknown'])[:5])
        st.markdown(f"**Authors:** {authors_str}")
        
        # Publication info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            journal_info = selected_paper.get('journal', 'N/A')
            # Add volume/issue/pages if available (for local papers)
            if selected_paper.get('volume') or selected_paper.get('issue') or selected_paper.get('pages'):
                journal_info += f", Vol {selected_paper.get('volume', '')}({selected_paper.get('issue', '')}), pp. {selected_paper.get('pages', '')}"
            st.markdown(f"**Journal:** {journal_info}")
            st.markdown(f"**Year:** {selected_paper.get('year', 'N/A')}")
        with col_info2:
            if selected_paper.get('citation_count'):
                st.metric("Citations", selected_paper['citation_count'])
            if selected_paper.get('doi'):
                st.markdown(f"**DOI:** {selected_paper['doi']}")
        
        st.divider()
        
        # Abstract
        st.markdown("**Abstract:**")
        st.write(selected_paper.get('abstract', 'No abstract available'))
        
        # Keywords/Fields of Study (if available)
        if selected_paper.get('fieldsOfStudy'):
            st.markdown(f"**Fields of Study:** {', '.join(selected_paper['fieldsOfStudy'])}")
        elif selected_paper.get('keywords'):
            st.markdown(f"**Keywords:** {', '.join(selected_paper['keywords'])}")
        
        # URL
        if selected_paper.get('url'):
            st.markdown(f"[🔗 View Paper]({selected_paper['url']})")
        
        st.divider()
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("➕ Add to List", key=f"add_{selected_paper.get('id')}", use_container_width=True, type="primary"):
                if selected_paper not in st.session_state.selected_papers:
                    st.session_state.selected_papers.append(selected_paper)
                    st.success("Added!")
                    st.rerun()
        
        with col_btn2:
            if st.button("🤖 Explain Relevance", key=f"explain_{selected_paper.get('id')}", use_container_width=True):
                with st.spinner("Generating explanation..."):
                    try:
                        # Use appropriate query based on data source
                        query_for_explanation = ""
                        if data_source == "Search Online":
                            query_for_explanation = st.session_state.get('last_search_query', '')
                        else:
                            query_for_explanation = st.session_state.get('local_search', '') or "research papers"
                        
                        explanation = explain_relevance(selected_paper, query_for_explanation)
                        st.session_state.ai_explanations[selected_paper.get('id')] = explanation
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # User Feedback Section
        st.subheader("💬 Feedback")
        
        paper_id = selected_paper.get('id')
        feedback_key = f"feedback_{paper_id}"
        
        if feedback_key not in st.session_state.paper_feedback:
            st.session_state.paper_feedback[feedback_key] = {"relevant": None, "note": ""}
        
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("✅ Relevant", key=f"rel_{paper_id}", use_container_width=True):
                st.session_state.paper_feedback[feedback_key]["relevant"] = True
                st.success("Marked as relevant!")
        with col_fb2:
            if st.button("❌ Not Relevant", key=f"notrel_{paper_id}", use_container_width=True):
                st.session_state.paper_feedback[feedback_key]["relevant"] = False
                st.info("Marked as not relevant")
        
        # Show current feedback status
        if st.session_state.paper_feedback[feedback_key]["relevant"] is not None:
            status = "✅ Relevant" if st.session_state.paper_feedback[feedback_key]["relevant"] else "❌ Not Relevant"
            st.caption(f"Status: {status}")
        
        # Note field
        note = st.text_area("Quick note (optional)", key=f"note_{paper_id}", height=80)
        if st.button("💾 Save Note", key=f"save_note_{paper_id}"):
            st.session_state.paper_feedback[feedback_key]["note"] = note
            st.success("Note saved!")
        
        st.divider()
        
        # AI Explanation
        st.subheader("🤖 AI Explanation")
        
        if paper_id in st.session_state.ai_explanations:
            st.info(st.session_state.ai_explanations[paper_id])
        else:
            st.caption("Click 'Explain Relevance' button above to get AI-generated relevance explanation")
    
    else:
        # Check if selected paper ID exists but doesn't belong to current data source
        if st.session_state.selected_paper_id and not selected_paper:
            # Clear the selected paper ID if it doesn't belong to current data source
            st.session_state.selected_paper_id = None
        
        if not all_papers:
            if data_source == "Local Papers":
                st.info("👈 Load local papers to view details")
            else:
                st.info("👈 Search for papers online to view details")
        else:
            st.info("👈 Select a paper from the results to view details")

# ==================== BOTTOM: READING LIST ====================
st.divider()
st.header("📌 Your Reading List")

if st.session_state.selected_papers:
    for idx, p in enumerate(st.session_state.selected_papers, 1):
        col_list1, col_list2 = st.columns([4, 1])
        with col_list1:
            st.markdown(f"{idx}. **{p['title']}** ({p.get('year', 'N/A')}) — {p.get('journal', 'N/A')}")
        with col_list2:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.selected_papers.remove(p)
                st.rerun()
else:
    st.caption("No papers in your reading list yet. Add papers using the 'Add to List' button.")

# ==================== FEEDBACK SUMMARY SECTION ====================
st.divider()
st.header("📊 Your Feedback Summary")

# Collect all papers with feedback
def get_papers_with_feedback():
    """Collect all papers that have been marked as relevant or not relevant"""
    relevant_papers = []
    not_relevant_papers = []
    
    # Use all loaded papers from session state
    all_available_papers = st.session_state.all_loaded_papers.copy()
    
    # Also add papers from cached searches
    for cached_papers_list in st.session_state.cached_papers.values():
        all_available_papers.extend(cached_papers_list)
    
    # Remove duplicates based on paper ID
    seen_ids = set()
    unique_papers = []
    for p in all_available_papers:
        paper_id = p.get('id', hash(p.get('title', '')))
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(p)
    
    # Check feedback for each paper
    for paper in unique_papers:
        paper_id = paper.get('id', hash(paper.get('title', '')))
        feedback_key = f"feedback_{paper_id}"
        
        if feedback_key in st.session_state.paper_feedback:
            feedback = st.session_state.paper_feedback[feedback_key]
            if feedback.get("relevant") is True:
                relevant_papers.append({
                    "paper": paper,
                    "note": feedback.get("note", "")
                })
            elif feedback.get("relevant") is False:
                not_relevant_papers.append({
                    "paper": paper,
                    "note": feedback.get("note", "")
                })
    
    return relevant_papers, not_relevant_papers

# Get papers with feedback
relevant_papers, not_relevant_papers = get_papers_with_feedback()

if relevant_papers or not_relevant_papers:
    tab_relevant, tab_not_relevant = st.tabs([
        f"✅ Relevant ({len(relevant_papers)})",
        f"❌ Not Relevant ({len(not_relevant_papers)})"
    ])
    
    with tab_relevant:
        if relevant_papers:
            st.caption(f"You've marked {len(relevant_papers)} paper(s) as relevant to your research.")
            for idx, item in enumerate(relevant_papers, 1):
                paper = item["paper"]
                note = item["note"]
                paper_id = paper.get('id', hash(paper.get('title', '')))
                
                with st.expander(f"{idx}. **{paper.get('title', 'Untitled')}**", expanded=False):
                    col_fb1, col_fb2 = st.columns([4, 1])
                    with col_fb1:
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown'])[:3])}")
                        st.markdown(f"**Journal:** {paper.get('journal', 'N/A')} • **Year:** {paper.get('year', 'N/A')}")
                        if note:
                            st.info(f"📝 **Your Note:** {note}")
                    with col_fb2:
                        if st.button("View", key=f"view_fb_rel_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
        else:
            st.info("No papers marked as relevant yet. Use the '✅ Relevant' button on paper details to mark papers.")
    
    with tab_not_relevant:
        if not_relevant_papers:
            st.caption(f"You've marked {len(not_relevant_papers)} paper(s) as not relevant.")
            for idx, item in enumerate(not_relevant_papers, 1):
                paper = item["paper"]
                note = item["note"]
                paper_id = paper.get('id', hash(paper.get('title', '')))
                
                with st.expander(f"{idx}. **{paper.get('title', 'Untitled')}**", expanded=False):
                    col_fb1, col_fb2 = st.columns([4, 1])
                    with col_fb1:
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown'])[:3])}")
                        st.markdown(f"**Journal:** {paper.get('journal', 'N/A')} • **Year:** {paper.get('year', 'N/A')}")
                        if note:
                            st.info(f"📝 **Your Note:** {note}")
                    with col_fb2:
                        if st.button("View", key=f"view_fb_notrel_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
        else:
            st.info("No papers marked as not relevant yet. Use the '❌ Not Relevant' button on paper details to mark papers.")
else:
    st.info("💡 **No feedback yet**\n\nMark papers as relevant or not relevant using the feedback buttons in the paper details panel to see them organized here.")
