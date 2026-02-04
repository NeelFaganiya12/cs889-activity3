import streamlit as st
import json
import pandas as pd
from typing import List, Dict
import requests
from datetime import datetime
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'selected_articles' not in st.session_state:
    st.session_state.selected_articles = []
if 'articles_data' not in st.session_state:
    st.session_state.articles_data = []

def load_json_articles(file_path: str) -> List[Dict]:
    """Load articles from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File {file_path} not found!")
        return []

def search_semantic_scholar(query: str, api_key: str = None, limit: int = 10) -> List[Dict]:
    """
    Search Semantic Scholar API for articles
    Note: Semantic Scholar API is free and doesn't require an API key for basic usage
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,venue,citationCount,url"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            papers = []
            for paper in data.get('data', []):
                papers.append({
                    "title": paper.get('title', 'N/A'),
                    "authors": [author.get('name', '') for author in paper.get('authors', [])],
                    "year": paper.get('year'),
                    "abstract": paper.get('abstract', 'No abstract available'),
                    "venue": paper.get('venue', 'N/A'),
                    "citations": paper.get('citationCount', 0),
                    "url": paper.get('url', ''),
                    "keywords": []  # Semantic Scholar doesn't provide keywords directly
                })
            return papers
        else:
            st.warning(f"Semantic Scholar API returned status {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching from Semantic Scholar: {str(e)}")
        return []

def initialize_gemini(api_key: str):
    """Initialize Gemini AI with API key"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def summarize_article(article: Dict, model) -> str:
    """Use Gemini to generate a concise summary of an article"""
    if not model:
        return "Gemini API not configured"
    
    try:
        prompt = f"""Please provide a concise summary (2-3 sentences) of this research paper:

Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'No abstract available')}

Summary:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def score_relevance(article: Dict, research_question: str, model) -> Dict:
    """Use Gemini to score article relevance to a research question"""
    if not model:
        return {"score": 0, "reasoning": "Gemini API not configured"}
    
    try:
        prompt = f"""Rate the relevance of this research paper to the following research question on a scale of 1-10, and provide a brief explanation:

Research Question: {research_question}

Paper Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'No abstract available')}

Please respond in this format:
Score: [1-10]
Reasoning: [brief explanation]"""
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse response
        score = 0
        reasoning = text
        if "Score:" in text:
            try:
                score = int(text.split("Score:")[1].split()[0])
            except:
                pass
        if "Reasoning:" in text:
            reasoning = text.split("Reasoning:")[1].strip()
        
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0, "reasoning": f"Error: {str(e)}"}

def filter_articles(articles: List[Dict], search_query: str) -> List[Dict]:
    """Filter articles based on search query"""
    if not search_query:
        return articles
    
    query_lower = search_query.lower()
    filtered = []
    for article in articles:
        # Search in title, abstract, authors, and keywords
        title_match = query_lower in article.get('title', '').lower()
        abstract_match = query_lower in article.get('abstract', '').lower()
        authors_match = any(query_lower in author.lower() for author in article.get('authors', []))
        keywords_match = any(query_lower in keyword.lower() for keyword in article.get('keywords', []))
        
        if title_match or abstract_match or authors_match or keywords_match:
            filtered.append(article)
    
    return filtered

def display_article(article: Dict, index: int, gemini_model=None, research_question: str = None):
    """Display a single article card"""
    with st.container():
        col1, col2 = st.columns([0.95, 0.05])
        
        with col1:
            st.markdown(f"### {article.get('title', 'Untitled')}")
            
            # Authors and metadata
            authors_str = ", ".join(article.get('authors', []))
            metadata = f"**Authors:** {authors_str} | **Year:** {article.get('year', 'N/A')} | **Venue:** {article.get('venue', 'N/A')} | **Citations:** {article.get('citations', 0)}"
            st.markdown(metadata)
            
            # AI-powered relevance score (if Gemini is configured and research question provided)
            if gemini_model and research_question:
                if f"relevance_{index}" not in st.session_state:
                    with st.spinner("Analyzing relevance..."):
                        relevance = score_relevance(article, research_question, gemini_model)
                        st.session_state[f"relevance_{index}"] = relevance
                else:
                    relevance = st.session_state[f"relevance_{index}"]
                
                score = relevance.get('score', 0)
                color = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
                st.markdown(f"{color} **Relevance Score:** {score}/10")
                with st.expander("See reasoning"):
                    st.write(relevance.get('reasoning', 'No reasoning available'))
            
            # Abstract
            abstract = article.get('abstract', 'No abstract available')
            st.markdown(f"**Abstract:** {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
            
            # AI summary button (if Gemini is configured)
            if gemini_model:
                if st.button(f"🤖 AI Summary", key=f"summary_btn_{index}"):
                    with st.spinner("Generating AI summary..."):
                        summary = summarize_article(article, gemini_model)
                        st.info(summary)
            
            # Keywords
            keywords = article.get('keywords', [])
            if keywords:
                keywords_str = " | ".join([f"`{kw}`" for kw in keywords])
                st.markdown(f"**Keywords:** {keywords_str}")
            
            # URL
            url = article.get('url', '')
            if url:
                st.markdown(f"[View Paper]({url})")
        
        with col2:
            is_selected = index in st.session_state.selected_articles
            if st.checkbox("Select", key=f"select_{index}", value=is_selected):
                if index not in st.session_state.selected_articles:
                    st.session_state.selected_articles.append(index)
            else:
                if index in st.session_state.selected_articles:
                    st.session_state.selected_articles.remove(index)
        
        st.divider()

# Main app
st.title("📚 Literature Review Assistant")
st.markdown("Search and select relevant articles for your literature review")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Local JSON File", "Semantic Scholar API"],
        help="Choose between local JSON file or live search via Semantic Scholar API"
    )
    
    # Gemini API key (for AI features)
    gemini_api_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        help="Enter your Gemini API key for AI-powered features"
    )
    
    gemini_model = None
    if gemini_api_key:
        gemini_model = initialize_gemini(gemini_api_key)
        if gemini_model:
            st.success("✅ Gemini API configured")
        else:
            st.error("❌ Failed to initialize Gemini")
    
    st.divider()
    
    # Filter options
    st.header("🔍 Filters")
    min_year = st.number_input("Minimum Year", value=2020, min_value=1900, max_value=2025)
    min_citations = st.number_input("Minimum Citations", value=0, min_value=0)
    
    st.divider()
    
    # Selected articles count
    st.header("📋 Selection")
    st.metric("Selected Articles", len(st.session_state.selected_articles))

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Search & Browse", "📋 Selected Articles", "📊 Statistics"])

with tab1:
    # Research question for AI relevance scoring
    research_question = None
    if gemini_model:
        research_question = st.text_input(
            "🎯 Research Question (for AI relevance scoring)",
            placeholder="Enter your research question to get AI-powered relevance scores...",
            help="Optional: Enter a research question to get AI-powered relevance scores for each article"
        )
    
    # Search bar
    search_query = st.text_input(
        "🔍 Search Articles",
        placeholder="Search by title, author, keywords, or abstract...",
        help="Enter keywords to search through articles"
    )
    
    # Load articles based on data source
    if data_source == "Local JSON File":
        if not st.session_state.articles_data or st.button("🔄 Reload Articles"):
            st.session_state.articles_data = load_json_articles("articles.json")
            st.success(f"Loaded {len(st.session_state.articles_data)} articles from JSON file")
    else:  # Semantic Scholar API
        if search_query:
            with st.spinner("Searching Semantic Scholar..."):
                st.session_state.articles_data = search_semantic_scholar(search_query, limit=20)
                if st.session_state.articles_data:
                    st.success(f"Found {len(st.session_state.articles_data)} articles")
        elif not st.session_state.articles_data:
            st.info("Enter a search query to fetch articles from Semantic Scholar API")
    
    # Filter articles
    if st.session_state.articles_data:
        # Create list of (index, article) tuples to preserve original indices
        articles_with_indices = [(i, article) for i, article in enumerate(st.session_state.articles_data)]
        
        # Filter by search query
        if search_query:
            filtered_with_indices = [
                (idx, article) for idx, article in articles_with_indices
                if article in filter_articles([article], search_query)
            ]
        else:
            filtered_with_indices = articles_with_indices
        
        # Apply additional filters
        filtered_with_indices = [
            (idx, article) for idx, article in filtered_with_indices
            if (article.get('year', 0) or 0) >= min_year and (article.get('citations', 0) or 0) >= min_citations
        ]
        
        st.subheader(f"Found {len(filtered_with_indices)} article(s)")
        
        # Display articles
        for original_idx, article in filtered_with_indices:
            display_article(article, original_idx, gemini_model, research_question)
    else:
        st.info("No articles loaded. Select a data source and search for articles.")

with tab2:
    st.header("Selected Articles")
    
    if st.session_state.selected_articles:
        selected_data = [st.session_state.articles_data[i] for i in st.session_state.selected_articles if i < len(st.session_state.articles_data)]
        
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_articles = []
            st.rerun()
        
        # Display selected articles
        for idx, article_idx in enumerate(st.session_state.selected_articles):
            if article_idx < len(st.session_state.articles_data):
                article = st.session_state.articles_data[article_idx]
                with st.expander(f"{idx + 1}. {article.get('title', 'Untitled')}", expanded=True):
                    st.markdown(f"**Authors:** {', '.join(article.get('authors', []))}")
                    st.markdown(f"**Year:** {article.get('year', 'N/A')} | **Venue:** {article.get('venue', 'N/A')} | **Citations:** {article.get('citations', 0)}")
                    st.markdown(f"**Abstract:** {article.get('abstract', 'No abstract available')}")
                    if article.get('url'):
                        st.markdown(f"[View Paper]({article.get('url')})")
        
        # Export options
        st.divider()
        st.subheader("📥 Export Selected Articles")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export to JSON"):
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "selected_count": len(selected_data),
                    "articles": selected_data
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="selected_articles.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export to CSV"):
                df = pd.DataFrame(selected_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="selected_articles.csv",
                    mime="text/csv"
                )
    else:
        st.info("No articles selected yet. Go to the Search & Browse tab to select articles.")

with tab3:
    st.header("📊 Statistics")
    
    if st.session_state.articles_data:
        df = pd.DataFrame(st.session_state.articles_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", len(st.session_state.articles_data))
        
        with col2:
            st.metric("Selected Articles", len(st.session_state.selected_articles))
        
        with col3:
            if 'year' in df.columns:
                avg_year = df['year'].mean()
                st.metric("Average Year", f"{avg_year:.0f}" if pd.notna(avg_year) else "N/A")
        
        with col4:
            if 'citations' in df.columns:
                total_citations = df['citations'].sum()
                st.metric("Total Citations", f"{total_citations:,}")
        
        # Charts
        if len(df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'year' in df.columns:
                    st.subheader("Articles by Year")
                    year_counts = df['year'].value_counts().sort_index()
                    st.bar_chart(year_counts)
            
            with col2:
                if 'citations' in df.columns:
                    st.subheader("Citations Distribution")
                    st.bar_chart(df['citations'].head(10))
        
        # Top venues
        if 'venue' in df.columns:
            st.subheader("Top Venues")
            venue_counts = df['venue'].value_counts().head(10)
            st.dataframe(venue_counts, use_container_width=True)
    else:
        st.info("Load articles to see statistics.")
