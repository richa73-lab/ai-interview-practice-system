from rag.pdf_loader import extract_text_from_pdf
from rag.embedder import chunk_text, create_embeddings, model
from rag.retriever import VectorStore
from llm.interviewer import generate_interview_question
from llm.evaluator import evaluate_answer

print("=== AI Interview Practice System ===")

# STEP 1 — Load PDF
text = extract_text_from_pdf("pdf.pdf")

# STEP 2 — Process PDF
chunks = chunk_text(text)
embeddings = create_embeddings(chunks)

vector_db = VectorStore(embeddings, chunks)

# STEP 3 — Ask topic from user
query = input("Enter topic you studied: ")

query_embedding = model.encode(query)
results = vector_db.search(query_embedding)

context = results[0]

# STEP 4 — Generate question
question = generate_interview_question(context)

print("\nAI Question:")
print(question)

# STEP 5 — Student answer
answer = input("\nYour Answer: ")


evaluation = evaluate_answer(question, answer)

print("\nAI Evaluation:")
print(evaluation)
