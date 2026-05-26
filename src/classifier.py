import re
from collections import Counter


def _match_count(patterns, text):
    count = 0
    for p in patterns:
        if isinstance(p, str):
            if p in text:
                count += 1
        else:
            # assume compiled regex
            if p.search(text):
                count += 1
    return count


def classify_document(text):
    """Classify a document into one of: Invoice, Resume, Utility Bill, Other, Unclassifiable.

    Uses a simple weighted keyword + regex scoring approach for robustness.
    """

    if not text or not text.strip():
        return "Other / Unclassifiable"

    lower = text.lower()

    # Safety short-circuit for very short docs
    if len(lower.strip()) < 30:
        return "Other / Unclassifiable"

    # Define keyword and regex hints for each class
    invoice_patterns = [
        "invoice",
        re.compile(r"invoice\s*#"),
        re.compile(r"invoice\s+no\.?"),
        "total amount",
        "bill to",
        re.compile(r"inv[- ]?\d+"),
    ]

    resume_patterns = [
        "experience",
        "education",
        "skills",
        "curriculum vitae",
        re.compile(r"\bresume\b"),
        re.compile(r"\b\d+\s+years?\b"),
    ]

    bill_patterns = [
        "amount due",
        "account number",
        "usage",
        "kwh",
        "meter reading",
        re.compile(r"amount\s+due"),
    ]

    # Score each category
    scores = Counter()

    scores["Invoice"] = _match_count(invoice_patterns, lower)
    scores["Resume"] = _match_count(resume_patterns, lower)
    scores["Utility Bill"] = _match_count(bill_patterns, lower)

    # Small boost if currency/amount-looking tokens present for invoice/bill
    if re.search(r"\$\s?\d|usd|usd\b|€|£|amount", lower):
        scores["Invoice"] += 1
        scores["Utility Bill"] += 1

    # Heuristic: resumes commonly contain sections like 'education' and 'skills'
    if "education" in lower and "skills" in lower:
        scores["Resume"] += 2

    # Choose best score
    most_common = scores.most_common()

    # If no hints at all
    if all(v == 0 for _, v in most_common):
        return "Other / Unclassifiable"

    # Get best and second best
    best, best_score = most_common[0]
    second_score = most_common[1][1] if len(most_common) > 1 else 0

    # If tie or low confidence, classify as Other
    if best_score == 0 or best_score == second_score:
        return "Other / Unclassifiable"

    return best