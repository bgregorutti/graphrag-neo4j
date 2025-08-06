from neo4j import GraphDatabase
import re
from sentence_transformers import SentenceTransformer
import json
from loguru import logger
from dotenv import dotenv_values


def parse(pdf_text):
    """
    Parse the given PDF text to extract the title, sections, subsections, and paragraphs.

    Args:
        pdf_text (str): The text content of the PDF.

    Returns:
        list: A list of dictionaries, each representing a section, subsection, or paragraph with its type, name, and content.
    """
    # Extract title
    title_match = re.search(r'Title: (.+)', pdf_text)
    title = title_match.group(1) if title_match else "No Title Found"

    # Extract sections, subsections, and paragraphs
    sections = re.findall(r'(\d{1,2}\.\s.+?)\n((?:[^2-9]+\n)*?)(?=\n\d{1,2}\.\s|$)', pdf_text, re.DOTALL)
    data = []
    for section in sections:
        section_title = section[0].strip()
        content = section[1].strip()

        # Add section node
        data.append({
            "type": "Section",
            "name": section_title,
            "content": content
        })

        # Extract subsections
        subsections = re.findall(r'(\d+\.\d+\s.+?)(?=\n\d+\.\d+|\n\d+\.\s|$)', content, re.DOTALL)
        for subsection in subsections:
            subsection_title = subsection.strip()
            data.append({
                "type": "Subsection",
                "name": subsection_title,
                "content": subsection_title
            })

        # Extract paragraphs (split by double newline)
        paragraphs = re.split(r'\n\n', content)
        for paragraph in paragraphs:
            if paragraph.strip():
                data.append({
                    "type": "Paragraph",
                    "name": paragraph.strip(),
                    "content": paragraph.strip()
                })
    return data

def add_node(tx, node_type, name, content=None):
    """
    Create a node of a given type with name and optional content in the Neo4j database.

    Args:
        tx: Neo4j transaction object.
        node_type (str): The type/label of the node.
        name (str): The name property of the node.
        content (str, optional): The content property of the node.
    """
    if content is not None:
        query = (
            f"CREATE (n:{node_type} {{name: $name, content: $content}})"
        )
        tx.run(query, name=name, content=content)
    else:
        query = (
            f"CREATE (n:{node_type} {{name: $name}})"
        )
        tx.run(query, name=name)

def add_node_with_embedding(tx, node_type, name, content=None):
    """
    Create a node with an embedding property in the Neo4j database.

    Args:
        tx: Neo4j transaction object.
        node_type (str): The type/label of the node.
        name (str): The name property of the node.
        content (str, optional): The content property of the node.
    """
    embedding = embedder.encode(content if content else name).tolist()
    if content is not None:
        query = (
            f"CREATE (n:{node_type} {{name: $name, content: $content, embedding: $embedding}})"
        )
        tx.run(query, name=name, content=content, embedding=embedding)
    else:
        query = (
            f"CREATE (n:{node_type} {{name: $name, embedding: $embedding}})"
        )
        tx.run(query, name=name, embedding=embedding)

def add_relationship(tx, start_node_type, start_node_name, end_node_type, end_node_name, relationship_type):
    """
    Create a relationship between two nodes of given types and names in the Neo4j database.

    Args:
        tx: Neo4j transaction object.
        start_node_type (str): Type/label of the start node.
        start_node_name (str): Name of the start node.
        end_node_type (str): Type/label of the end node.
        end_node_name (str): Name of the end node.
        relationship_type (str): Type of the relationship.
    """
    query = (
        f"MATCH (a:{start_node_type} {{name: $start_name}}), "
        f"(b:{end_node_type} {{name: $end_name}}) "
        f"CREATE (a)-[:{relationship_type}]->(b)"
    )
    tx.run(query, start_name=start_node_name, end_name=end_node_name)

def delete_all(tx):
    """
    Delete all nodes and relationships in the Neo4j database.

    Args:
        tx: Neo4j transaction object.
    """
    tx.run("MATCH (n) DETACH DELETE n")

def store_graph(example, session):
    """
    Store nodes and relationships from the example data into the Neo4j database.

    Args:
        example (list): List of dictionaries representing nodes.
        session: Neo4j session object.
    """
    logger.info("Storing nodes")
    for item in example:
        session.execute_write(add_node_with_embedding, item["type"], item["name"], item["content"])
        
    logger.info("Storing relationships")
    for pos, item in enumerate(example):
        related = [subitem for subitem in example[pos+1:] if subitem["type"] != item["type"] and subitem["related_to"] == item["name"]]

        for related_item in related:
            session.execute_write(add_relationship, item["type"], item["name"], related_item["type"], related_item["name"], "contains")
            session.execute_write(add_relationship, related_item["type"], related_item["name"], item["type"], item["name"], "is_part_of")

def query_paragraphs(question, driver):
    """
    Query paragraphs related to a given question from the Neo4j database.

    Args:
        question (str): The section name to search for.
        driver: Neo4j driver object.

    Returns:
        list: List of paragraph contents related to the section.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paragraph)-[:is_part_of]->(s:Section)
            WHERE s.name CONTAINS $section_name
            RETURN p.content AS paragraph_content
            """,
            section_name=question
        )
        
        paragraphs = [record["paragraph_content"] for record in result]
    return paragraphs

if __name__ == "__main__":
    """
    Main execution block for loading example data, connecting to Neo4j, and storing the graph.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open("example.json", "r") as f:
        example = json.load(f)

    # Connect to the Neo4j database
    ENVS = {**dotenv_values(".env")}
    driver = GraphDatabase.driver(ENVS.get("URI"), auth=(ENVS.get("USER"), ENVS.get("PASSWORD")))

    with driver.session() as session:
        logger.info("Deleting all existing nodes and relationships")
        session.execute_write(delete_all)
        
        logger.info("Storing graph in Neo4j")
        store_graph(example, session)
        

    # print(query_paragraphs("Methodology", driver))

    driver.close()
