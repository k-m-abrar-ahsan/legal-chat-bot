import json
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ---------------------------------------------------------
# Load source JSON
# ---------------------------------------------------------
with open("text.json") as f:
    data = json.load(f)

# ---------------------------------------------------------
# Configuration
# Long sections (over this char count) get split into
# clause-level chunks so no single topic gets diluted.
# ---------------------------------------------------------
SPLIT_THRESHOLD = 800  # chars — sections longer than this get clause-split

# ---------------------------------------------------------
# Helper: extract individual lettered clauses from a
# sub-section string like:
#   "...(a) no driver shall drive under alcohol... (b) no conductor..."
# Returns a list of clause strings, or the original string
# if no lettered clauses are found.
# ---------------------------------------------------------
def extract_clauses(text):
    # Split on lettered clause markers: (a), (b), (c) ...
    parts = re.split(r'\s*\(([a-z])\)\s*', text)
    # re.split with a capture group gives: [before, letter, content, letter, content ...]
    # Zip letters with their content
    clauses = []
    if len(parts) < 3:
        # No lettered clauses found — return as single item
        return [text.strip()]

    # parts[0] is the preamble before first clause
    preamble = parts[0].strip()
    # Walk through letter, content pairs
    i = 1
    while i + 1 < len(parts):
        letter = parts[i]
        content = parts[i + 1].strip()
        if content:
            clauses.append(f"({letter}) {content}")
        i += 2

    return clauses if clauses else [text.strip()]

# ---------------------------------------------------------
# Build chunks
#
# Strategy:
#   SHORT sections  → one chunk for the whole section
#   LONG sections   → one chunk per clause, each carrying
#                     the section title + chapter as context
#
# Penalty sections (Chapter XI) always get the rule section
# they reference appended to their chunk, so the embedding
# captures both the punishment AND the behaviour it covers.
# ---------------------------------------------------------

# First pass: index all sections by number for cross-ref lookup
section_index = {}
for chapter in data["hierarchy"]:
    for section in chapter["sections"]:
        section_index[section["section_number"]] = {
            "chapter": chapter["chapter_title"],
            "section": section,
        }

chunks = []

for chapter in data["hierarchy"]:
    for section in chapter["sections"]:
        sec_num    = section["section_number"]
        sec_title  = section["section_title"]
        ch_title   = chapter["chapter_title"]
        content    = section["content"]
        content_str = json.dumps(content)

        # Find any rule sections that THIS section references
        # (relevant for penalty sections like "Penalty for s49")
        referenced_nums = re.findall(r'section[s]?\s+(\d+)', sec_title.lower())
        referenced_text = ""
        for ref_num in referenced_nums:
            if ref_num in section_index and ref_num != sec_num:
                ref_sec = section_index[ref_num]["section"]
                referenced_text += (
                    f"\n\nReferenced rule — Section {ref_num}: "
                    f"{ref_sec['section_title']}\n"
                    + json.dumps(ref_sec["content"])[:600]
                )

        # --- Decide: whole-section chunk or clause-level chunks ---
        if len(content_str) <= SPLIT_THRESHOLD:
            # SHORT: one chunk for the whole section
            rich_text = (
                f"Chapter: {ch_title}\n"
                f"Section {sec_num}: {sec_title}\n"
                f"Content: {content_str}"
                + referenced_text
            )
            chunks.append({
                "section_number": sec_num,
                "section_title":  sec_title,
                "chapter":        ch_title,
                "clause":         None,          # whole section
                "rich_text":      rich_text,
                "raw_content":    section,       # full section returned to LLM
            })

        else:
            # LONG: emit one chunk per clause so each topic
            # gets its own focused embedding
            for sub_key, sub_text in content.items():
                if not isinstance(sub_text, str):
                    # e.g. definitions is a list — join it
                    if isinstance(sub_text, list):
                        sub_text = " ".join(sub_text)
                    else:
                        sub_text = json.dumps(sub_text)

                clauses = extract_clauses(sub_text)

                for clause_text in clauses:
                    if len(clause_text.strip()) < 20:
                        continue  # skip empty/boilerplate fragments

                    rich_text = (
                        f"Chapter: {ch_title}\n"
                        f"Section {sec_num}: {sec_title}\n"
                        f"Clause: {clause_text}"
                        + referenced_text
                    )
                    chunks.append({
                        "section_number": sec_num,
                        "section_title":  sec_title,
                        "chapter":        ch_title,
                        "clause":         clause_text,   # the specific clause
                        "rich_text":      rich_text,
                        "raw_content":    section,       # full section returned to LLM
                    })

print(f"Built {len(chunks)} chunks from 126 sections")
print(f"  (clause-level splitting expanded long sections into focused embeddings)\n")

# Show a sample of what section 49 produced
s49_chunks = [c for c in chunks if c["section_number"] == "49"]
print(f"Section 49 produced {len(s49_chunks)} chunks:")
for c in s49_chunks[:5]:
    print(f"  - {c['clause'][:80] if c['clause'] else '[whole section]'}")
print("  ...")
print()

# ---------------------------------------------------------
# Embed all chunks
# ---------------------------------------------------------
print("Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Embedding {len(chunks)} chunks...")
texts      = [c["rich_text"] for c in chunks]
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# ---------------------------------------------------------
# Build FAISS index (cosine similarity via inner product
# on L2-normalised vectors)
# ---------------------------------------------------------
dimension = embeddings.shape[1]  # 384 for MiniLM
index     = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# ---------------------------------------------------------
# Save to disk
# ---------------------------------------------------------
faiss.write_index(index, "legal_index.faiss")
with open("legal_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"\n✅ Done!")
print(f"   Chunks embedded : {len(chunks)}")
print(f"   legal_index.faiss — saved")
print(f"   legal_chunks.pkl  — saved")
print(f"\nRun: python main.py")