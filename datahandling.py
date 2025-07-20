from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import dotenv
dotenv.load_dotenv(override=True)

# Neo4j Connection Info
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load multilingual embedding model (XLM-RoBERTa based)
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

# Step 1: Extract FAQ questions
'''def fetch_questions(tx):
    result = tx.run("MATCH (f:FAQ) RETURN id(f) AS id, 
    f.question AS question")
    return [(record["id"], record["question"]) for record in result]

# Step 2: Store embeddings back into Neo4j
def store_embedding(tx, node_id, embedding):
    tx.run("""
        MATCH (f:FAQ) WHERE id(f) = $id
        SET f.vector = $embedding
    """, id=node_id, embedding=embedding)

# Main process
with driver.session() as session:
    faqs = session.read_transaction(fetch_questions)

    for node_id, question in faqs:
        embedding = model.encode(question).tolist()
        session.write_transaction(store_embedding, node_id, embedding)

driver.close()'''

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

embedding = model.encode("Test sentence")
print(len(embedding)) 
