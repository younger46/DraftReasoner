"""MechAgent prompts for the ReAct loop + the answer judge."""
from __future__ import annotations


ANSWER_TEMPLATE = """<think>{reasoning}</think>
<answer>{answer}</answer>"""


JUDGE_PROMPT = """You are a strict judge for a mechanical-drawing VQA. Decide if the model
answer is semantically correct (same key value/meaning) as the correct answer.
Question: {question}
Correct answer: {correct}
Model answer: {model}
Return 1 if correct or basically correct with all key info, else 0. Return only a JSON object with a 0/1 `score`.
"""


ANNOTATION_PROMPT = (
    "Look at this mechanical drawing. Extract the explicit dimension and annotation values that are "
    "visible. Return ONLY a JSON object with this shape: "
    '{"annotations": [{"value": "...", "feature": "...", "view": "...", '
    '"kind": "dimension|datum|tolerance|roughness|chamfer|count"}]}. '
    "List every distinct dimension/annotation you can read (e.g. values like Phi61, Phi51, 3-Phi10.5, "
    "Ra0.8, C1, tolerance). Do not invent values you cannot see. No prose, JSON only."
)


VIEW_PROMPT = (
    "Look at this mechanical drawing. Identify the views it contains and, if there are section views, "
    "which drawing view each corresponds to. Return ONLY a JSON object: "
    '{"views": [{"name": "front|top|side|section|isometric|local|detail", "note": "..."}], '
    '"section_map": [{"section": "A-A", "location": "front view"}]}. No prose, JSON only.'
)


REACT_SYSTEM = (
    "You are an expert mechanical engineer inspecting a mechanical drawing. You have tools that can "
    "read exact dimensions/annotations (AnnotationExtract), identify views (ViewAlign), split a "
    "composite sheet (FigureParse), compute from a dimension chain (GeometrySolve) and look up GB/T "
    "standards (StdKB). Reason about the question; if a tool would give a precise answer, call it, "
    "examine its JSON result, and continue until you can answer. When ready, give the final answer in "
    "the question's language, wrapped in <answer>...</answer>. Prefer using GeometrySolve (OCR) over "
    "guessing dimension values."
)
