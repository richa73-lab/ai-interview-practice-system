import streamlit as st
import tempfile

from rag.pdf_loader import extract_text_from_pdf
from rag.embedder import chunk_text, create_embeddings, model
from rag.retriever import VectorStore
from llm.interviewer import generate_interview_question
from llm.evaluator import evaluate_answer


st.title("🎤 AI Interview Practice System")
st.write("Upload your study PDF and practice interview questions.")


# ---------- PDF Upload ----------
uploaded_file = st.file_uploader("Upload your study PDF", type="pdf")

if uploaded_file:

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded!")

    # process PDF only once
    if "vector_db" not in st.session_state:

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)

        st.session_state.vector_db = VectorStore(embeddings, chunks)

    vector_db = st.session_state.vector_db

    # ---------- Topic ----------
    topic = st.text_input("Enter topic you studied")

    if topic:

        # retrieve relevant chunks
        query_embedding = model.encode(topic)
        results = vector_db.search(query_embedding)

        # ---------- Choose number of questions ----------
        num_questions = st.number_input(
            "How many questions you want to practice?",
            min_value=1,
            max_value=10,
            value=3
        )

        # ---------- Initialize Session State ----------
        if "question_count" not in st.session_state:
            st.session_state.question_count = 0

        if "current_question" not in st.session_state:
            st.session_state.current_question = ""

        if "evaluation" not in st.session_state:
            st.session_state.evaluation = ""

        if "contexts" not in st.session_state:
            st.session_state.contexts = []

        if "context_index" not in st.session_state:
            st.session_state.context_index = 0

        # ---------- Start Practice ----------
        if st.button("Start Practice"):

            st.session_state.question_count = 1
            st.session_state.context_index = 0

            # store top contexts (for diversity)
            st.session_state.contexts = results[:5]

            # generate first question
            context = st.session_state.contexts[0]
            st.session_state.current_question = generate_interview_question(context)
            st.session_state.evaluation = ""

        # ---------- Show Question ----------
        if st.session_state.question_count > 0:

            st.subheader(
                f"Question {st.session_state.question_count} / {num_questions}"
            )

            st.write(st.session_state.current_question)

            # ---------- Show Source from PDF ----------
            st.subheader("📄 Source from your PDF")
            current_context = st.session_state.contexts[
                st.session_state.context_index
            ]
            st.write(current_context)

            # ---------- Student Answer ----------
            answer = st.text_area("Your Answer")

            # ---------- Submit Answer ----------
            if st.button("Submit Answer") and answer:

                evaluation = evaluate_answer(
                    st.session_state.current_question,
                    answer
                )

                st.session_state.evaluation = evaluation

            # show evaluation
            if st.session_state.evaluation:

                st.subheader("AI Evaluation")
                st.write(st.session_state.evaluation)

                # ---------- Next Question ----------
                if st.session_state.question_count < num_questions:

                    if st.button("Next Question"):

                        st.session_state.question_count += 1

                        # rotate to next context
                        st.session_state.context_index += 1

                        if st.session_state.context_index >= len(st.session_state.contexts):
                            st.session_state.context_index = 0

                        next_context = st.session_state.contexts[
                            st.session_state.context_index
                        ]

                        st.session_state.current_question = generate_interview_question(
                            next_context
                        )

                        st.session_state.evaluation = ""
                        st.rerun()

                else:
                    st.success("🎉 Practice session completed!")
