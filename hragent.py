from operator import add
from typing import Annotated, List
import dotenv
from typing_extensions import TypedDict
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Annotated
from langgraph.graph import END
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph
import numpy as np


model =  SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')


def embed_text(text):
    return model.encode(text).tolist()

dotenv.load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

enhanced_graph = Neo4jGraph(enhanced_schema=True)

class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]
    history: List[Dict[str, str]]
    related_hr_data: List[str]
    answer: List[str]

class OutputState(TypedDict):
    question: str
    answer: str
    steps: List[str]
    cypher_statement: str


def format_history(history):
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

def find_related_hr_data(query_text, top_k=5, threshold=0.1):
    try:
        if len(query_text) == 0:
            return "Please enter your HR question"

        question_embedding_vector = model.encode(query_text)
        if not isinstance(question_embedding_vector, list):
            question_embedding_vector = question_embedding_vector.tolist()

        cypher_query = '''CALL db.index.vector.queryNodes('document_vector_index', $topK, $embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN  
        node.answer AS answer, 
        node.question AS question, 
        score
        ORDER BY score DESC
        LIMIT 10
        '''

        params = {
            "embedding": question_embedding_vector,
            "topK": int(top_k),
            "threshold": float(threshold)
        }

        result = enhanced_graph.query(cypher_query, params=params)
        print("HR related data:", result)

        result = result
        all_results_sorted = sorted(result, key=lambda r: r['score'], reverse=True)
        return all_results_sorted

    except Exception as e:
        print(f"Error in finding related HR data: {e}")
        return "Error"

def classify_hr_question(question):
    related_hr_data = find_related_hr_data(question)
    return related_hr_data

guardrails_system = """As an intelligent HR assistant, your primary objective is to decide whether a given question is related to HR policies, procedures, employee benefits, or workplace matters stored in the neo4j schema. 
If the question is related to HR topics in the neo4j schema, output "related". Otherwise, output "end".
If the question is a greeting or general workplace inquiry, allow that and provide a friendly response.
To make this decision, assess the content of the question and determine if it matches with HR-related neo4j schema, 
or related HR topics. Provide only the specified output: "related" or "end".
Do not generate syntax errors."""

guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        ( 
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("""Neo4j Schema: {enhanced_grap} 
Question: {question}"""),
        ),
    ]
)

class GuardrailsOutput(BaseModel):
    decision: Literal["related", "end"] = Field(
        description="Decision on whether the question is related to our HR database. Check against our HR schema"
    )

def log_output(output: GuardrailsOutput) -> GuardrailsOutput:
    return GuardrailsOutput(output)

guardrails_chain = guardrails_prompt | log_output | llm.with_structured_output(GuardrailsOutput)

def guardrails(state: OverallState) -> OverallState:
    related_hr_data = classify_hr_question(state["question"])
    state["related_hr_data"] = related_hr_data

    print(related_hr_data, "related_hr_data")   

    if related_hr_data is None or len(related_hr_data) == 0:
        decision = "end"
    else:
        decision = "related"    

    return {
        "question": state["question"],
        "schema": enhanced_graph,
        "next_action": decision,
        "related_hr_data": related_hr_data
    }

def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "related":
        return "generate_final_answer"

generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a highly trained HR Assistant for a leading organization. Your role is to respond clearly, helpfully, and efficiently to employees using HR data provided from internal systems.

            Always follow these principles:

            Be Employee-Focused: Treat every interaction with empathy and professionalism. Make employees feel heard, respected, and supported in their workplace queries.

            Answer Clearly and Directly: If the HR database provides sufficient information, respond with a concise, definitive answer. Use professional but approachable language.

            Ask for Exactly What's Missing:
            If the data is incomplete or unclear, do not ask general follow-ups like "Can you clarify?"
            Instead, prompt for specific details needed to proceed (e.g., "Could you provide your employee ID?" or "Which policy are you referring to?").

            End with Actionable Guidance:
            Close each response with a clear next step, suggestion, or resource for the employee (e.g., "You can find the complete policy document in the employee portal..." or "Please contact HR directly at [contact] for personalized assistance").

            Output guidelines:
            Write in professional, friendly language that employees can understand easily.
            Be natural and supportive.
            Avoid internal codes, technical jargon, or system language.
            Use the HR data retrieved to its fullest. If no useful data is found, politely guide the employee to provide exactly what's needed next.
            Maintain confidentiality and direct sensitive matters to appropriate HR personnel.

            Question: {question} 
            Results: {results}"""
        ),
        (
            "human",
            (
                """Question asked by the employee:

Respond as if you are answering the HR question directly.

Results: {results}

Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()
    
def generate_final_answer(state: OverallState) -> OutputState:
    """
    Generates final HR answer based on Neo4j database query results.
    """
    database_records = state.get("database_records")
    related_hr_data = state.get("related_hr_data")

    if database_records is None:
        database_records = related_hr_data
    else:
        top_match = max(related_hr_data, key=lambda x: x.get("score", 0))
        if top_match.get("score", 0) > 0.79:
            database_records = database_records + [top_match]

    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"), "results": database_records}
    )
    
    print("Final HR data:", database_records)
    history = state["history"]
    if history is None:
        history = []
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": final_answer})
    state["history"] = history
    print("Printing History:", history)
    return {"answer": final_answer, "steps": ["generate_final_answer"]}

def guardrails_condition(
    state: OverallState,
) -> Literal["generate_final_answer"]:
    return "generate_final_answer"

# Build the LangGraph workflow
langgraph = StateGraph(OverallState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition,
)
langgraph.add_edge("generate_final_answer", END)

langgraph = langgraph.compile()

print(langgraph.get_graph().draw_mermaid())

def start_chatbot(question):
    """
    Start the HR chatbot with an employee question.
    
    Args:
        question (str): Employee's HR-related question
        
    Returns:
        dict: Response from the HR agent
    """
    response = langgraph.invoke({"question": question, "history": []}) 
    print("HR Agent Response:", response)
    return response

# Example usage
if __name__ == "__main__":
    # Test the HR chatbot
    test_question = "What are the company's vacation policies?"
    result = start_chatbot(test_question)
    print("\nFinal Answer:", result.get("answer", "No answer provided"))