import json
import pandas as pd
from langchain_chroma import Chroma  # Updated import to use langchain-chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize the SentenceTransformerEmbeddings model
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load JSON data from file
f = open(r'C:\Users\sunb3\Documents\RAG\data.json', 'r')
data = json.load(f)

# Convert JSON to DataFrames for each metric
lead_time_df = pd.DataFrame(data['lead_time_stats']).T
mean_time_to_restore_df = pd.DataFrame(data['mean_time_to_restore_stats']).T
change_failure_rate_df = pd.DataFrame(data['change_failure_rate_stats']).T
deployment_frequency_df = pd.DataFrame(data['deployment_frequency_stats']).T
prs_df = pd.DataFrame(data['lead_time_prs'])

# Create documents for embeddings
docs = [
    Document(
        page_content=f"PR {pr['number']} titled '{pr['title']}' by {pr['author']['username']}",
        metadata=pr
    )
    for pr in data['lead_time_prs']
]

# Extract texts for embedding
texts = [doc.page_content for doc in docs]

# Generate embeddings for each document
embeddings = embedding_model.embed_documents(texts)

# Manually simplify metadata
def simplify_metadata(metadata):
    simplified = {}
    for key, value in metadata.items():
        # Only keep simple types
        if isinstance(value, (str, int, float, bool)):
            simplified[key] = value
        elif isinstance(value, dict):
            # Flatten nested dictionaries
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (str, int, float, bool)):
                    simplified[f"{key}_{subkey}"] = subvalue
    return simplified

simplified_metadata = [simplify_metadata(doc.metadata) for doc in docs]

# Initialize Chroma vector store with the embeddings function
vectorstore = Chroma(embedding_function=embedding_model)

# Add texts and embeddings to the vector store
vectorstore.add_texts(texts, metadatas=simplified_metadata, embeddings=embeddings)

# Define a function to query the RAG
def query_rag(question: str) -> str:
    llm = Ollama()
    context = vectorstore.similarity_search(question, k=5)
    context_texts = [doc.page_content for doc in context]  # Extract context texts
    # Concatenate context texts into a single prompt
    prompt = f"Context: {' '.join(context_texts)}\nQuestion: {question}"
    answer = llm.generate([prompt])
    return answer

# Example questions
questions = [
    "How are my 3 teams i.e. Prometheus, Kubernetes and Grafana doing according to the data? Use their four keys i.e. Deployment Frequency, Lead Time for Changes, Change Failure Rate and Time to Restore Service to formulate your answer.",
    "Which team is struggling: Prometheus, Kubernetes and Grafana, in terms of lead time and can we say why?",
    "What's working in the other teams that the struggling team from above could learn from?",
    "Are there any teams that are struggling with deployments? What are the current bottlenecks in our development pipeline?",
    "Which teams need additional support or resources? Highlight teams with high workloads, frequent overtime, or declining performance metrics.",
    "How is the overall deployment frequency across all teams?",
    "Which pull requests across Prometheus, Kubernetes and Grafana are taking the longest to merge, and can we say why?",
    "How does our current change failure rate compare to last week?",
    "Are there any patterns in first response times that could indicate potential bottlenecks in the review process?"
    
]

# Query the RAG system and print answers
for question in questions:
    answer = query_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {answer} ")
