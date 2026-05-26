import re
from datetime import datetime


# Helper regexes
_RE_DATE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")
_RE_DATE_DMY = re.compile(r"(\d{2}/\d{2}/\d{4})")
_RE_MONTH_NAME = re.compile(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b\s+\d{1,2},?\s+\d{4})", re.IGNORECASE)
_RE_CURRENCY = re.compile(r"[\$€£]\s?([\d,]+\.?\d*)")
_RE_NUMBER = re.compile(r"([\d,]+\.?\d*)")


def _parse_amount(m):
    if not m:
        return None
    s = m.group(1)
    s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return None


def _first_nonempty_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


# =========================
# INVOICE EXTRACTION
# =========================
def extract_invoice(text):
    data = {}

    # Invoice number: many possible labels
    invoice = re.search(
        r"(?:Invoice\s*(?:No\.?|Number|#)\s*[:\-]?\s*|\bINV[- ]?)([A-Za-z0-9-]*\d[A-Za-z0-9-]*)",
        text,
        re.IGNORECASE,
    )

    # Date: try multiple formats
    date = _RE_DATE_ISO.search(text) or _RE_DATE_DMY.search(text) or _RE_MONTH_NAME.search(text)

    # Company: look for common labels
    company = re.search(
        r"(?:Company|From|Bill To|Vendor|Supplier)[:\s]+([A-Za-z0-9 .,&'\-]+)",
        text,
        re.IGNORECASE,
    )

    # Amount: total, balance, invoice total
    amount = re.search(r"(?:Total\s+Amount|Invoice\s+Total|Total)[:\s]*[\$€£]?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    if not amount:
        amount = _RE_CURRENCY.search(text)

    data["invoice_number"] = invoice.group(1).strip() if invoice else None

    data["date"] = date.group(0).strip() if date else None

    data["company"] = company.group(1).strip() if company else None

    data["total_amount"] = _parse_amount(amount) if amount else None

    return data


# =========================
# RESUME EXTRACTION
# =========================
def extract_resume(text):
    data = {}

    # Name: first non-empty line that looks like a name (contains letters and spaces, not an email)
    first_line = _first_nonempty_line(text)
    name = None
    if first_line and '@' not in first_line and len(first_line.split()) <= 5:
        name = first_line

    # Email
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)

    # Phone: allow various formats
    phone = re.search(r'(?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.]*)?\d{3,4}[\s\-\.]?\d{3,4}', text)

    # Experience: look for first numeric years mention
    experience = re.search(r'(\d+)\+?\s*(?:years|yrs)\b', text, re.IGNORECASE)

    data["name"] = name

    data["email"] = email.group(0) if email else None

    data["phone"] = phone.group(0).strip() if phone else None

    data["experience_years"] = int(experience.group(1)) if experience else None

    return data


# =========================
# UTILITY BILL EXTRACTION
# =========================
def extract_bill(text):
    data = {}

    account = re.search(
        r'(?:Account\s*Number|Account)[:\s#-]*([A-Za-z0-9-]*\d[A-Za-z0-9-]*)',
        text,
        re.IGNORECASE,
    )

    # Date
    date = _RE_DATE_ISO.search(text) or _RE_DATE_DMY.search(text) or _RE_MONTH_NAME.search(text)

    # Usage: look for kwh or units
    usage = re.search(
        r'(?:Usage|Consumption)[:\s]*(?:KWH|kWh|kwh)?\s*([\d,]+)',
        text,
        re.IGNORECASE,
    )

    # Amount due: various labels
    amount = re.search(r'(?:Amount\s*Due|Balance\s*Due|Total\s*Due|Total)[:\s]*[\$€£]?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if not amount:
        amount = _RE_CURRENCY.search(text)

    data["account_number"] = account.group(1) if account else None

    data["date"] = date.group(0).strip() if date else None

    data["usage_kwh"] = int(usage.group(1).replace(',', '')) if usage else None

    data["amount_due"] = _parse_amount(amount) if amount else None

    return data