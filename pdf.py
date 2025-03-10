import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup
import textwrap
import pandas as pd

# Optional: Add wordcloud if available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Optional: Add transformers if available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(layout="wide")  # Enable wide mode

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    
    .page-display {
        background-image: url('https://source.unsplash.com/800x600/?book,paper');
        background-size: cover;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    
    /* Add more styling for better readability */
    .answer-section {
        background-color: rgba(0, 255, 170, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .source-section {
        background-color: rgba(100, 100, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    /* Highlight key terms */
    .highlight {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 2px 5px;
        border-radius: 3px;
    }
    
    /* Improve readability of text */
    p, li {
        line-height: 1.6;
        font-size: 16px;
    }
    
    h3 {
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    /* Style for the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E2129;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00FFAA !important;
        color: #0E1117 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced Prompt Template for better structured responses
IMPROVED_PROMPT_TEMPLATE = """
You are an expert research assistant specializing in document analysis.

CONTEXT INFORMATION:
{document_context}

USER QUESTION: 
{user_query}

Provide a comprehensive answer based ONLY on the information in the context.
Format your response in these sections:
1. Direct Answer (1-2 sentences with the core answer)
2. Key Details (2-3 bullet points with supporting information)
3. Source Context (brief mention of where this information appears in the document)

If the answer cannot be determined from the context, clearly state that.
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

K = 3  # Number of top clusters to return

def save_uploaded_file(uploaded_file):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)




def web_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if "url?q=" in href and not "webcache" in href:
            results.append(href.split("url?q=")[1].split("&")[0])
        if len(results) >= 5:
            break
    return results

def cluster_documents(documents, k=3):
    """Cluster documents and return top representative documents from each cluster"""
    if not documents or len(documents) <= k:
        return documents
    
    # Extract text content from documents
    texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=min(k, len(texts)), random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Find documents closest to cluster centers
    centers = kmeans.cluster_centers_
    top_docs = []
    
    for i in range(len(centers)):
        cluster_docs_indices = np.where(clusters == i)[0]
        if len(cluster_docs_indices) > 0:
            # Calculate distances to center
            distances = cosine_similarity(tfidf_matrix[cluster_docs_indices], centers[i].reshape(1, -1))
            # Get index of document closest to center
            closest_idx = cluster_docs_indices[np.argmax(distances)]
            top_docs.append(documents[closest_idx])
    
    return top_docs

def generate_answer(user_query, context_documents):
    """Generate a more accurate and structured answer based on the context"""
    
    # Extract relevant text from documents
    context_text = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in context_documents])
    
    # Create a conversation prompt with our improved template
    conversation_prompt = ChatPromptTemplate.from_template(IMPROVED_PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    
    # Get the raw response
    raw_response = response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text
    })
    
    return raw_response

def extract_key_points(text):
    """Extract key points from text using simple rules if transformers not available"""
    # Simple rule-based extraction of first sentence of each paragraph
    paragraphs = text.split('\n')
    key_points = []
    
    for para in paragraphs:
        if para.strip():
            sentences = para.split('. ')
            if sentences:
                # Get first sentence with period added back
                first_sentence = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
                key_points.append(first_sentence)
    
    # Return the top 3 key points
    return '\n\n'.join(key_points[:3])

def visualize_knowledge_graph(documents):
    """Create a knowledge graph visualization from the documents with improved term extraction and relationships."""
    G = nx.Graph()
    
    # Combine all document content
    combined_text = " ".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents])
    
    # Use TF-IDF to extract important terms
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)  # Limit to top 50 terms
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create a dictionary of terms and their TF-IDF scores
    term_scores = dict(zip(feature_names, tfidf_scores))
    
    # Filter out terms with low TF-IDF scores
    filtered_terms = [term for term, score in term_scores.items() if score > 0.1]
    
    # Split text into sentences for co-occurrence analysis
    sentences = combined_text.split('. ')
    
    # Build co-occurrence relationships
    for sentence in sentences:
        sentence_terms = [term for term in filtered_terms if term in sentence.lower()]
        for i in range(len(sentence_terms) - 1):
            for j in range(i + 1, len(sentence_terms)):
                term1 = sentence_terms[i]
                term2 = sentence_terms[j]
                if G.has_edge(term1, term2):
                    G[term1][term2]['weight'] += 1
                else:
                    G.add_edge(term1, term2, weight=1)
    
    # Limit graph size for better visualization
    if len(G.nodes()) > 30:
        # Keep only the most connected nodes
        sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]
        nodes_to_keep = [n for n, d in sorted_nodes]
        G = G.subgraph(nodes_to_keep)
    
    # Create a layout for the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust k for better spacing
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=[G.degree(n) * 300 for n in G.nodes()],  # Scale node size by degree
        node_color='skyblue', 
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, pos, 
        width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()],  # Scale edge width by weight
        edge_color='gray', 
        alpha=0.6
    )
    
    nx.draw_networkx_labels(
        G, pos, 
        font_size=12, 
        font_weight='bold', 
        font_color='black'
    )
    
    plt.title("Knowledge Graph of Key Terms", fontsize=16)
    plt.axis("off")
    st.pyplot(plt)
    return G

def create_wordcloud(documents):
    """Create a word cloud from the documents if wordcloud is available"""
    if not WORDCLOUD_AVAILABLE:
        return None
        
    combined_text = " ".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents])
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='#0E1117',
        colormap='viridis',
        contour_color='#00FFAA',
        contour_width=1,
        max_words=100
    ).generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

def display_page_summary(query, context_documents, ai_response):
    """Create a more structured and visually appealing display of results"""
    
    # Create tabs to organize information
    tabs = st.tabs(["AI Answer", "Source Context", "Visualization"])
    
    with tabs[0]:
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown("### Answer")
        st.markdown(ai_response)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Extract key terms for highlighting
        if TRANSFORMERS_AVAILABLE:
            try:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                key_points = summarizer(ai_response, max_length=100, min_length=30, do_sample=False)
                if key_points:
                    st.markdown("### Key Points")
                    st.markdown(key_points[0]['summary_text'])
            except Exception:
                key_points = extract_key_points(ai_response)
                st.markdown("### Key Points")
                st.markdown(key_points)
        else:
            key_points = extract_key_points(ai_response)
            st.markdown("### Key Points")
            st.markdown(key_points)
    
    with tabs[1]:
        st.markdown("### Source Context")
        
        for i, doc in enumerate(context_documents):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # Limit content length for display
            if len(content) > 500:
                content = content[:497] + "..."
            
            with st.expander(f"Source {i+1}", expanded=i==0):
                st.markdown(f'<div class="source-section">', unsafe_allow_html=True)
                st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Try to extract page numbers or metadata
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.markdown("**Metadata:**")
                    for key, value in doc.metadata.items():
                        st.markdown(f"- {key}: {value}")
    
    with tabs[2]:
        st.markdown("### Knowledge Graph")
        G = visualize_knowledge_graph(context_documents)
        
        # Add a word cloud if possible
        if WORDCLOUD_AVAILABLE:
            st.markdown("### Word Cloud")
            wc_plot = create_wordcloud(context_documents)
            if wc_plot:
                st.pyplot(wc_plot)
        
        # Add document similarity matrix
        st.markdown("### Document Similarity")
        texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context_documents]
        if len(texts) > 1:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create a DataFrame for better visualization
            sim_df = pd.DataFrame(
                similarity_matrix,
                index=[f"Doc {i+1}" for i in range(len(texts))],
                columns=[f"Doc {i+1}" for i in range(len(texts))]
            )
            
            st.dataframe(sim_df.style.background_gradient(cmap='viridis'))

# UI Layout
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Upload Document")
    uploaded_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )
    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        st.success("✅ Document processed successfully! Ask your questions below.")
        
    # Add settings section
    with st.expander("⚙️ Settings"):
        response_type = st.radio(
            "Response Format:",
            ["Concise", "Detailed", "Academic"],
            index=1
        )
        
        if response_type == "Concise":
            IMPROVED_PROMPT_TEMPLATE = """
            You are an expert research assistant. Use the provided context to answer the query.
            Be extremely concise (max 2 sentences). Focus only on direct facts.
            
            Query: {user_query} 
            Context: {document_context} 
            Answer:
            """
        elif response_type == "Academic":
            IMPROVED_PROMPT_TEMPLATE = """
            You are an academic research assistant with expertise in document analysis.
            
            CONTEXT INFORMATION:
            {document_context}
            
            RESEARCH QUESTION: 
            {user_query}
            
            Provide a comprehensive academic response with:
            1. Executive Summary (2-3 sentences)
            2. Key Findings (3-5 bullet points with citations to source material)
            3. Critical Analysis (evaluate strengths/limitations of the information)
            4. References (indicate specific sections of the source material)
            
            Use formal academic language and maintain scholarly rigor.
            """

with col2:
    st.markdown("#### Ask a Question")
    user_input = st.chat_input("Enter your question about the document...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            top_docs = cluster_documents(relevant_docs, k=K)
            ai_response = generate_answer(user_input, top_docs)
        
        with st.chat_message("assistant", avatar="🤖"):
            display_page_summary(user_input, top_docs, ai_response)
