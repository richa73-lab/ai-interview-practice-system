from rag.pdf_loader import extract_text_from_pdf
from rag.embedder import chunk_text, create_embeddings, model
from rag.retriever import VectorStore
from llm.interviewer import generate_interview_question

# load PDF
text = extract_text_from_pdf("pdf.pdf")

# chunk + embed
chunks = chunk_text(text)
embeddings = create_embeddings(chunks)

# create vector DB
vector_db = VectorStore(embeddings, chunks)

# ask topic
query = "Normalization in DBMS"

query_embedding = model.encode(query)
results = vector_db.search(query_embedding)

context = results[0]

print("Retrieved Context:\n", context)

# generate interview question
question = generate_interview_question(context)

print("\nGenerated Interview Question:\n", question)
