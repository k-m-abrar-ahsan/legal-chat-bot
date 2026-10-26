import json
import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------
# 1. Load Vector Index (built by setup.py)
# ---------------------------------------------------------
print("Loading vector index...")
index = faiss.read_index("legal_index.faiss")
with open("legal_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✅ Index loaded ({len(chunks)} sections)\n")

# ---------------------------------------------------------
# 2. Load LLM
# ---------------------------------------------------------
MODEL_ID = "models/gemma-2-9b-it"  # Change this if you downloaded to a custom path
print("⏳ Loading Qwen into RTX 4060 Ti...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
print("✅ Model loaded!\n")

# ---------------------------------------------------------
# Helper: LLM inference
# ---------------------------------------------------------
def generate_hf_response(system_prompt, user_prompt, temperature=0.1, max_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )
    new_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

# ---------------------------------------------------------
# Step 0: Query Expansion
# Rewrites the user's casual question into formal legal
# language BEFORE embedding, so the vector search finds
# the right sections even when vocabulary doesn't match.
# ---------------------------------------------------------
def expand_query(user_question):
    print("[0] Expanding query into legal language...")

    system_prompt = """You are a legal language translator for Bangladesh's Road Transport Act, 2018.

Rewrite the user's question using formal legal terms that would appear in the actual law text.

CRITICAL RULE: Always lead with the BEHAVIOUR or SUBJECT first, penalty terms last.
BAD:  "Is imprisonment imposed on individuals found driving under the influence of alcohol?"
GOOD: "driving under the influence of alcohol motor vehicle — penalty imprisonment fine"

BAD:  "What punishment is given for operating without a fitness certificate?"
GOOD: "motor vehicle fitness certificate expired unfit operation — penalty punishment"

TRANSLATION RULES:
- helmet                     → "protective headgear motorcycle safety equipment"
- drunk driving / booze      → "driving under the influence of alcohol intoxicant motor vehicle"
- broken lights / headlights → "signalling lights equipment layout fitness certificate construction motor vehicle"
- jail / locked up           → "imprisonment"
- fine / penalty             → "fine imprisonment penalty taka"
- expired licence            → "expired driving license"
- driving without licence    → "driving without driving license prohibited"
- company / corporate crime  → "company director owner offence committed by company"
- overloading                → "overloading weight limit goods vehicle"
- route permit               → "route permit transport vehicle"
- accident / crash           → "accident compensation third party"
- speeding                   → "speed limit reckless driving"
- registration               → "registration certificate motor vehicle"
- fitness / roadworthy       → "fitness certificate fitness of motor vehicle"
- demerit points             → "demerit points allocation of points driving license"
- rickshaw                   → "three-wheeled vehicle non-motorized vehicle"
- foreigner                  → "foreign national driving license"
- cops / police              → "police officer"
- confiscate / seize         → "seize seizure documents motor vehicle"
- import tax / customs       → "out of scope — not covered by Road Transport Act"
- age / minimum age / how old   → "minimum age eighteen years driving license eligibility conditions"
- 16-year-old / 17-year-old     → "minimum age eighteen years driving license eligibility conditions section 6"
- professional licence / bus driver licence → "professional driving license twenty one years conditions section 6"
- qualifications / requirements → "conditions for driving license issuance section 6"

Output ONLY the rewritten phrase. No question marks. No explanation."""

    expanded = generate_hf_response(system_prompt, user_question, temperature=0.1, max_tokens=128)

    # Fallback: if model returns something malformed, use original
    if len(expanded) > len(user_question) * 5 or len(expanded) < 5:
        print("    Expansion failed — using original question")
        return user_question

    print(f"    Original : {user_question}")
    print(f"    Expanded : {expanded}")
    return expanded

# ---------------------------------------------------------
# Step 1: Semantic Retrieval
# Embeds the EXPANDED query and finds the top-k most
# semantically similar sections from the FAISS index.
# Works for any question phrasing — no keywords needed.
# ---------------------------------------------------------
# In retrieve_sections(), replace the single search with this:

def retrieve_sections(user_question, expanded_question, top_k=4):
    print("[1] Semantic search across all sections...")

    behaviour_query = expanded_question
    penalty_query = f"penalty fine imprisonment offence punishment for: {expanded_question}"

    seen_sections = set()  # deduplicate by section NUMBER, not index position
    results = []

    for label, query in [("rule", behaviour_query), ("penalty", penalty_query)]:
        query_vec = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        for score, idx in zip(scores[0], indices[0]):
            chunk = chunks[idx]
            sec_num = chunk["section_number"]

            if sec_num in seen_sections:
                continue
            seen_sections.add(sec_num)

            results.append({
                "score":          float(score),
                "section_number": sec_num,
                "section_title":  chunk["section_title"],
                "chapter":        chunk["chapter"],
                "content":        chunk["raw_content"]["content"],
            })
            print(f"    [{label}:{score:.3f}] Section {sec_num}: {chunk['section_title']}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results
# ---------------------------------------------------------
# Step 2: Answer Generation
# LLM receives the ORIGINAL user question + retrieved
# sections and generates a plain-English answer.
# ---------------------------------------------------------
def generate_answer(user_question, retrieved_sections):
    print("[2] Generating answer...")

    context = "\n\n---\n\n".join([
        f"Section {s['section_number']} ({s['chapter']}): {s['section_title']}\n"
        + json.dumps(s["content"], indent=2)
        for s in retrieved_sections
    ])

    system_prompt = f"""You are a legal assistant specializing in Bangladesh's Road Transport Act, 2018.

Answer the user's question using ONLY the legal text provided below.

RULES:
1. If clearly answered: respond in plain English and cite the section number.
2. If partially answered: answer what you can, then state what the law does not specify.
3. If not in this Act: say "This is not covered by the Road Transport Act, 2018" and suggest where to look.
4. If a section says penalties or details are "prescribed by rules": say exactly this —
   "The Act recognises this but the specific [amount/points/fee] is set in subordinate rules not included here."
5. NEVER invent specific numbers (fines, months, points, ages, fees) that are not explicitly
   stated in the retrieved text. If a number is not in the text, say it is not specified here.
6. CRITICAL: Do not confuse a starting balance with a deduction. If the law says
   "12 points are allotted" that is the total balance, not the deduction per offence.
7. English only.

RETRIEVED LEGAL SECTIONS:
{context}"""

    return generate_hf_response(system_prompt, user_question, temperature=0.2, max_tokens=512)

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🏛️  Bangladesh Road Transport Act — Legal RAG")
    print("      (Semantic Search + Query Expansion)")
    print("=" * 50 + "\n")

    while True:
        user_input = input("Ask a legal question (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        # Step 0: rewrite question in legal vocabulary for better embedding match
        expanded = expand_query(user_input)

        # Step 1: find relevant sections using the expanded query
        sections = retrieve_sections(user_input, expanded, top_k=4)

        # Step 2: generate answer using the ORIGINAL question + retrieved sections
        answer = generate_answer(user_input, sections)

        print(f"\n🤖 LEGAL ADVISOR:\n{answer}\n")
        print("-" * 50)