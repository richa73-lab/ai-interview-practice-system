from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def evaluate_answer(question, student_answer):

    prompt = f"""
You are a technical interviewer.

Question: {question}

Student Answer: {student_answer}

Evaluate the answer strictly.

Give output in this format:

Correctness: Good / Average / Poor
Score: x/10
Feedback: one short improvement suggestion
Ideal Answer: short correct answer

Evaluation:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
