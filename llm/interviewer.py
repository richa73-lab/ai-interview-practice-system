# from transformers import pipeline

# # load model
# generator = pipeline(
#     "text-generation",
#     model="distilgpt2"
# )


# def generate_interview_question(context):

#     # stronger instruction to AI
#     prompt = f"""
# Generate ONE short interview question about the following database topic.

# Topic:
# {context}

# Question:
# """

#     try:
#         result = generator(
#             prompt,
#             max_new_tokens=20,   # smaller output
#             temperature=0.2,     # less randomness
#             do_sample=True
#         )

#         full_text = result[0]["generated_text"]

#         # remove prompt part, keep only generated question
#         question = full_text.split("Question:")[-1].strip()

#         return question

#     except Exception as e:
#         return f"Model Error: {str(e)}"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load better model
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_interview_question(context):

    prompt = f"""
You are a strict technical interviewer.

Generate ONE interview question ONLY from the study material below.

Rules:
- Use ONLY the information in the study material
- Do NOT use external knowledge
- If question cannot be formed from material, say "No question possible"

Study Material:
{context}

Interview Question:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=30
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
