from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from google.api_core.exceptions import ResourceExhausted, NotFound
from config import MODEL_ORDER

def build_plain_prompt():
    template = (
        """
You are an assistant that answers using ONLY the provided context.

Provide a concise single-line direct answer first. If the answer is not present in the context, respond exactly with:
Answer is not available in the context.

After that, provide a short explanation in 1-3 sentences. Keep paragraphs short.

Context:
{context}

Question:
{question}
"""
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

def build_bullets_prompt():
    template = (
        """
You are an assistant that answers using ONLY the provided context.

Start with a single-line direct answer (or "Answer is not available in the context."). Then include a 'Key points:' section with 3-6 bullet points listing the important facts or steps.

Context:
{context}

Question:
{question}
"""
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_answer_with_fallback_using_prompt(prompt_template: PromptTemplate, docs, question):
    """Try models from MODEL_ORDER and return (text, model_name, error_or_none)."""
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
            response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

            # normalize response shapes
            text = None
            if isinstance(response, dict):
                for key in ("output_text", "text", "answer", "output"):
                    if key in response and response[key]:
                        text = response[key]
                        break
                if not text:
                    for v in response.values():
                        if isinstance(v, str) and v.strip():
                            text = v
                            break
                        if isinstance(v, list) and v:
                            parts = [p for p in v if isinstance(p, str) and p.strip()]
                            if parts:
                                text = "\n".join(parts)
                                break
            elif isinstance(response, str):
                text = response
            else:
                try:
                    text = str(response)
                except Exception:
                    text = None

            if text and text.strip():
                return text, model_name, None
            else:
                # try next model
                continue

        except ResourceExhausted:
            continue
        except NotFound:
            continue
        except Exception:
            continue

    return None, None, "All models failed or exhausted their quotas."
