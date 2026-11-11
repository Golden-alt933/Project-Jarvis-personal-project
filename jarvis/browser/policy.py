"""What the browser tools may do without asking.

Ordinary browsing (reading, navigating, scrolling, clicking, typing) is safe.
This module decides the exceptions from what the extension reports about the
target element (see ``describe`` in the extension):

- ``ask``: a consequential action (purchase, send, delete, sign-in, ...). The
  tool asks the user itself, after resolving the element.
- ``refuse``: credential and payment fields. The user must type those.

The classifier errs towards asking: any label it cannot place in the safe
default of "just navigating or editing" but that looks like a commitment asks.
Everything here works on strings from the page, so it is pure and unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse


@dataclass(frozen=True)
class Verdict:
    kind: Literal["safe", "ask", "refuse"]
    reason: str = ""

    @property
    def safe(self) -> bool:
        return self.kind == "safe"


SAFE = Verdict("safe")


def _ask(reason: str) -> Verdict:
    return Verdict("ask", reason)


def _norm(text: Any) -> str:
    """Lowercase, split camelCase, turn separators into spaces."""
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text or ""))
    value = re.sub(r"[_\-./:]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


# -- credential and payment fields ---------------------------------------------
_AUTOCOMPLETE_CREDENTIAL = re.compile(r"\b(cc |one time code|current password|new password)")
_CREDENTIAL_WORDS = re.compile(
    r"\b("
    r"password|passwords|passcode|passwd|pwd|cvv|cvc|cvn|csc|"
    r"card ?(?:number|no|num|holder)|credit card|debit card|cc ?(?:num|number)|name on card|"
    r"security code|otp|one ?time (?:pass)?code|verification code|auth(?:entication)? code|"
    r"2fa|mfa|authenticator|expir(?:y|ation)(?: ?(?:date|month|year))?|"
    r"pin(?! ?code)|upi pin|atm pin|account number|iban|routing number|"
    r"ssn|social security|aadhaar|aadhar"
    r")\b"
)
_FIELD_ATTRS = ("label", "ariaLabel", "name", "id", "placeholder", "title")


def credential_reason(element: dict[str, Any]) -> str | None:
    """Why this field must be filled in by the user, or None if it may be typed into."""
    if str(element.get("type", "")).lower() == "password":
        return "a password field"
    if _AUTOCOMPLETE_CREDENTIAL.search(_norm(element.get("autocomplete"))):
        return "a credential or payment field"
    for attr in _FIELD_ATTRS:
        found = _CREDENTIAL_WORDS.search(_norm(element.get(attr)))
        if found:
            return f"a credential or payment field ('{found.group(0)}')"
    return None


def looks_like_card_number(text: str) -> bool:
    """13-19 digits (spaces and dashes allowed) that pass the Luhn check."""
    digits = re.sub(r"[ -]", "", str(text))
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        n = int(char)
        if index % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def classify_typing(element: dict[str, Any], text: str = "") -> Verdict:
    reason = credential_reason(element)
    if reason:
        return Verdict("refuse", f"This is {reason}. The user must enter it themselves.")
    if looks_like_card_number(text):
        return Verdict("refuse", "That text looks like a payment card number. The user must enter it themselves.")
    return SAFE


# -- consequential clicks ------------------------------------------------------
_SAFE_PHRASES = re.compile(
    r"\b(?:add(?:ed)? to (?:my )?(?:cart|bag|basket|wish ?list|list|favou?rites|queue|playlist|watch ?list)|"
    r"remove from (?:cart|bag|basket|wish ?list|list|favou?rites|queue|playlist|watch ?list)|"
    r"save for later|continue shopping|sort by|filter by|clear filters?)\b"
)
_DANGER: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a purchase or payment",
        re.compile(
            r"\b(buy|purchase|purchasing|place (?:your |my |the )?order|order now|pay|paying|payment|"
            r"paypal|gpay|donate|donation|subscribe|subscription|upgrade|book now|reserve|rent|bid|"
            r"start (?:a |your )?(?:free )?trial|confirm (?:order|booking|reservation|purchase|payment)|"
            r"complete (?:order|purchase|booking|payment))\b"
        ),
    ),
    (
        "sending or publishing something",
        re.compile(r"\b(send|post|publish|tweet|reply|share|invite|broadcast|submit)\b"),
    ),
    (
        "deleting something",
        re.compile(
            r"\b(delete|remove|erase|wipe|destroy|discard|deactivate|terminate|trash|unsubscribe|"
            r"clear (?:all|history|data|cookies|browsing)|"
            r"cancel (?:my |the |your )?(?:order|subscription|account|membership|booking|reservation|plan|trip))\b"
        ),
    ),
    (
        "an account or security change",
        re.compile(
            r"\b(passwords?|passcode|two ?factor|2fa|mfa|security|api keys?|access tokens?|"
            r"recovery (?:email|phone|codes?)|change (?:my |your )?(?:email|phone|username|name|plan)|"
            r"update (?:my |your )?(?:email|phone|billing|payment|address)|"
            r"(?:link|connect) (?:an? )?account|disconnect|revoke|log ?out (?:of )?all|sign out everywhere)\b"
        ),
    ),
    (
        "signing in or authorising access",
        re.compile(
            r"\b(sign ?in|log ?in|sign ?up|register|create (?:an? |your )?account|"
            r"continue with (?:google|apple|facebook|microsoft|github|email)|"
            r"authori[sz]e|authori[sz]ation|allow|grant|approve|verify|oauth)\b"
        ),
    ),
    ("accepting terms", re.compile(r"\b(accept|agree|acknowledge)\b")),
    (
        "moving money",
        re.compile(r"\b(transfer|withdraw|withdrawal|top ?up|recharge|redeem|wire|refund|deposit)\b"),
    ),
    ("a confirmation", re.compile(r"\bconfirm\b")),
)
_AMOUNT = re.compile(r"(?:[₹$€£¥]\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:rs|inr|usd|eur|gbp|rupees|dollars)\b|\brs\.? ?\d)")
_AMBIGUOUS_LABELS = frozenset(
    {"ok", "okay", "yes", "go", "done", "proceed", "finish", "complete", "apply", "save", "save changes",
     "update", "continue", "next step", "i understand"}
)
_TERMS_CHECKBOX = re.compile(r"\b(terms|conditions|consent|privacy policy|policy)\b")
_RISKY_PATH = re.compile(
    r"(?:^|[/_\-.])(checkout|payments?|pay|purchase|buy|delete|remove|password|security|login|signin|"
    r"signup|register|oauth|authorize|subscribe|donate|transfer|unsubscribe)(?:$|[/_\-.])"
)
_BUTTON_TYPES = {"submit", "button", "image", "reset"}


def _texts(element: dict[str, Any]) -> str:
    buttonish = element.get("tag") == "button" or str(element.get("type", "")).lower() in _BUTTON_TYPES
    parts = [element.get("label"), element.get("text"), element.get("ariaLabel"), element.get("title")]
    if buttonish:
        parts += [element.get("value"), element.get("name"), element.get("id")]
    seen: list[str] = []
    for part in parts:
        text = _norm(part)
        if text and text not in seen:
            seen.append(text)
    return " | ".join(seen)


_SUBMIT_WORD = re.compile(r"\bsubmit\b")


def _is_link(element: dict[str, Any]) -> bool:
    return element.get("tag") == "a" and element.get("role", "") in ("", "link")


def _risky_path(url: str) -> bool:
    if not url:
        return False
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        return True  # unparseable: unsure, so ask
    return bool(_RISKY_PATH.search(path))


def classify_click(element: dict[str, Any]) -> Verdict:
    """Decide whether clicking this element needs the user's approval."""
    kind = str(element.get("type", "")).lower()
    if kind == "file":
        return _ask("choosing a file to upload")
    text = _texts(element)
    stripped = _SAFE_PHRASES.sub(" ", text)
    if "search" in stripped:
        # A search form's own button is often named like "nav-search-submit-button". Running a
        # search sends nothing; every other danger word on it still counts.
        stripped = _SUBMIT_WORD.sub(" ", stripped)
    for reason, pattern in _DANGER:
        if pattern.search(stripped):
            return _ask(reason)
    if kind in ("checkbox", "radio") or element.get("role") in ("checkbox", "switch"):
        if _TERMS_CHECKBOX.search(text):
            return _ask("accepting terms or consent")
    link = _is_link(element)
    if not link and stripped == text and _AMOUNT.search(text):
        return _ask("a purchase or payment (the control shows a price)")
    if _risky_path(str(element.get("href") or "")):
        return _ask("a page that handles payment, sign-in or account changes")
    submit_like = kind in ("submit", "image") and not link
    form = element.get("form") or {}
    if submit_like and _risky_path(str(form.get("action") or "")):
        return _ask("submitting a form to a payment, sign-in or account page")
    if not link and text.strip(" |.!…") in _AMBIGUOUS_LABELS:
        return _ask("a generic confirm button whose effect is unclear")
    if submit_like and not text:
        return _ask("an unlabeled submit button")
    return SAFE


def classify_submit(element: dict[str, Any]) -> Verdict:
    """Decide whether pressing Enter in this field (submitting its form) needs approval."""
    form = element.get("form")
    if not form:
        searchy = "search" in _texts(element) or str(element.get("type", "")).lower() == "search"
        if searchy:
            return SAFE
        return _ask("pressing Enter here may send a message or submit something")
    if _risky_path(str(form.get("action") or "")):
        return _ask("submitting a form to a payment, sign-in or account page")
    for submit_label in form.get("submitLabels") or []:
        verdict = classify_click({"tag": "button", "type": "submit", "label": submit_label})
        if not verdict.safe:
            return _ask(f"submitting a form whose button says '{submit_label}' ({verdict.reason})")
    return SAFE
