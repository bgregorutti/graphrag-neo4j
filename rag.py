from neo4j import GraphDatabase
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import dotenv_values

def graph_rag(question, driver, generator, top_k=5):
    """
    Implements the Graph Retrieval-Augmented Generation (RAG) workflow.

    Given a question, retrieves relevant nodes and their context from a Neo4j graph database,
    constructs a prompt, and generates an answer using a text generation model.

    Args:
        question (str): The input question.
        driver: The Neo4j database driver.
        generator: The text generation pipeline/model.
        top_k (int): Number of top relevant nodes to retrieve.

    Returns:
        str: The generated answer.
    """
    retrieved_content = retrieve_relevant_nodes_with_graph(question, driver, top_k=top_k)
    context = "\n".join(retrieved_content)
    prompt = f"Question: {question}\nContext:\n{context}\nAnswer:"
    print(f"Prompt: {prompt}")
    
    # Generate response using the model
    response = generator(prompt, max_length=100)
    
    # Return the generated answer
    return response[0]['generated_text']

def retrieve_relevant_nodes(question, driver, top_k=5):
    """
    Retrieves the top-k nodes from the Neo4j database most relevant to the question,
    based on embedding similarity.

    Args:
        question (str): The input question.
        driver: The Neo4j database driver.
        top_k (int): Number of top relevant nodes to retrieve.

    Returns:
        list: List of tuples (similarity, node name, node content) for the top-k nodes.
    """
    question_emb = embedder.encode(question)
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE exists(n.embedding) RETURN n.name AS name, n.content AS content, n.embedding AS embedding"
        )
        nodes = []
        for record in result:
            emb = np.array(record["embedding"])
            sim = np.dot(question_emb, emb) / (np.linalg.norm(question_emb) * np.linalg.norm(emb))
            nodes.append((sim, record["name"], record["content"]))
        nodes.sort(reverse=True)
        return nodes[:top_k]

def retrieve_relevant_nodes_with_graph(question, driver, top_k=5):
    """
    Retrieves the top-k relevant nodes based on embedding similarity and expands the context
    by traversing the graph to include related nodes (e.g., parents, children).

    Args:
        question (str): The input question.
        driver: The Neo4j database driver.
        top_k (int): Number of top relevant nodes to retrieve.

    Returns:
        list: List of content strings from the top nodes and their related nodes.
    """
    question_emb = embedder.encode(question)
    with driver.session() as session:
        # Step 1: Retrieve top-k nodes by embedding similarity
        result = session.run(
            "MATCH (n) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.content AS content, n.embedding AS embedding, elementId(n) AS node_id"
        )

        scored_nodes = []
        for record in result:
            emb = np.array(record["embedding"])
            sim = np.dot(question_emb, emb) / (np.linalg.norm(question_emb) * np.linalg.norm(emb))
            scored_nodes.append((sim, record["node_id"], record["name"], record["content"]))
        scored_nodes.sort(reverse=True)
        top_nodes = scored_nodes[:top_k]
        node_ids = [n[1] for n in top_nodes]

        # Step 2: Traverse the graph to get related nodes (e.g., parents, children)
        related = session.run(
            """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            OPTIONAL MATCH (n)-[:is_part_of|contains*1..2]-(related)
            RETURN collect(DISTINCT related.content) AS related_contents
            """,
            node_ids=node_ids
        )
        related_contents = []
        for record in related:
            related_contents.extend([c for c in record["related_contents"] if c])
        # Combine top node contents and related contents
        contents = [n[3] for n in top_nodes if n[3]] + related_contents
        return contents

if __name__ == "__main__":
    """
    Example usage of the Graph RAG pipeline.

    Loads environment variables, connects to the Neo4j database, initializes the embedding
    and text generation models, and runs a sample question through the RAG workflow.
    """
    # Connect to the Neo4j database
    ENVS = {**dotenv_values(".env")}
    driver = GraphDatabase.driver(ENVS.get("URI"), auth=(ENVS.get("USER"), ENVS.get("PASSWORD")))

    # Initialize the SentenceTransformer for embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize the generative model pipeline
    generator = pipeline("text-generation", model="openai-community/gpt2")


    # Example usage
    question = "What are the key findings from the introduction section?"
    answer = graph_rag(question, driver, generator, 5)
    print(answer)

    driver.close()
