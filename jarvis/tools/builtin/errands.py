"""Errands: prefilled email, calendar, map and message pages, ready for the user to review.

Each tool builds a correct, fully escaped link (or an .ics file) and opens it, in the
JARVIS tab group of the user's Chrome when the extension is connected (so Jarvis can then
read the page and carry on, for example press Save after the user agrees) or in the
default browser or app otherwise. Nothing here sends, saves or books anything: the user
reviews the page, and pressing the final button through the browser tools asks first.

The URL builders are plain functions (tested without a browser); the tools around them
only parse the arguments and open the result.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.parse import urlparse

from jarvis import platform as plat
from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.builtin.productivity import WhenError, parse_when
from jarvis.tools.paths import known_folder

log = logging.getLogger(__name__)

DEFAULT_EVENT_MINUTES = 60
_ADDRESS = re.compile(r"^[^@\s,;<>\"]+@[^@\s,;<>\"]+\.[^@\s,;<>\"]+$")
_SPLIT = re.compile(r"[,;\n]+")


class ErrandError(ValueError):
    """The arguments cannot make a safe, meaningful link."""


# -- small helpers -----------------------------------------------------------------------------
def enc(text: str, safe: str = "") -> str:
    """Percent-encode for a URL component: spaces as %20, newlines as %0A, '+' as %2B."""
    return quote(text, safe=safe)


def _query(pairs: list[tuple[str, str]], safe: dict[str, str] | None = None) -> str:
    safe = safe or {}
    return "&".join(f"{k}={enc(v, safe.get(k, ''))}" for k, v in pairs if v != "")


def addresses(text: str | list[str] | None, label: str = "recipient") -> list[str]:
    """Split 'a@x.com, b@y.com; c@z.org' into validated addresses. Raises ErrandError."""
    if text is None:
        return []
    parts = text if isinstance(text, list) else [text]
    found: list[str] = []
    for part in parts:
        for item in _SPLIT.split(str(part)):
            item = item.strip()
            if not item:
                continue
            m = re.search(r"<([^<>]+)>\s*$", item)  # "Name <a@x.com>"
            item = m.group(1).strip() if m else item
            if not _ADDRESS.match(item):
                raise ErrandError(f"'{item[:60]}' is not a valid email address")
            found.append(item)
    return found


def _oneline(text: str | None) -> str:
    return " ".join((text or "").split())


# -- email ------------------------------------------------------------------------------------------
def email_url(provider: str, to: list[str], subject: str, body: str, cc: list[str], bcc: list[str]) -> str:
    """The compose link for gmail, outlook or mailto."""
    subject = _oneline(subject)
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    recipients = lambda xs: ",".join(xs)  # noqa: E731
    if provider == "gmail":
        q = _query(
            [("view", "cm"), ("fs", "1"), ("to", recipients(to)), ("cc", recipients(cc)), ("bcc", recipients(bcc)),
             ("su", subject), ("body", body)],
            safe={"to": "@,", "cc": "@,", "bcc": "@,"},
        )
        return f"https://mail.google.com/mail/?{q}"
    if provider == "outlook":
        q = _query(
            [("to", recipients(to)), ("cc", recipients(cc)), ("bcc", recipients(bcc)), ("subject", subject), ("body", body)],
            safe={"to": "@,", "cc": "@,", "bcc": "@,"},
        )
        return f"https://outlook.office.com/mail/deeplink/compose?{q}"
    if provider == "mailto":
        # RFC 6068: spaces are %20 (never '+') and line breaks are CRLF.
        q = _query(
            [("cc", recipients(cc)), ("bcc", recipients(bcc)), ("subject", subject), ("body", body.replace("\n", "\r\n"))],
            safe={"cc": "@,", "bcc": "@,"},
        )
        head = "mailto:" + enc(recipients(to), safe="@,")
        return f"{head}?{q}" if q else head
    raise ErrandError(f"unknown mail provider '{provider}'")


# -- calendar ------------------------------------------------------------------------------------------
def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calendar_url(provider: str, title: str, start: datetime, end: datetime, location: str | None,
                 details: str | None, guests: list[str]) -> str:
    """The prefilled new-event link for google or outlook. Times go out in UTC ('Z'), which
    both services show in the user's own calendar zone, so no zone name can be wrong."""
    if provider == "google":
        q = _query(
            [("action", "TEMPLATE"), ("text", _oneline(title)), ("dates", f"{_utc_stamp(start)}/{_utc_stamp(end)}"),
             ("details", details or ""), ("location", _oneline(location)), ("add", ",".join(guests))],
            safe={"add": "@,"},
        )
        return f"https://calendar.google.com/calendar/render?{q}"
    if provider == "outlook":
        q = _query(
            [("path", "/calendar/action/compose"), ("rru", "addevent"), ("subject", _oneline(title)),
             ("startdt", _utc_iso(start)), ("enddt", _utc_iso(end)), ("body", details or ""),
             ("location", _oneline(location)), ("to", ",".join(guests))],
            safe={"path": "/", "to": "@,", "startdt": ":", "enddt": ":"},
        )
        return f"https://outlook.office.com/calendar/0/deeplink/compose?{q}"
    raise ErrandError(f"unknown calendar provider '{provider}'")


def _ics_text(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n"))


def _fold(line: str) -> list[str]:
    """RFC 5545 folding: no line longer than 75 octets, continuation lines start with a space."""
    out: list[str] = []
    current = ""
    for ch in line:
        limit = 75 if not out else 74  # the leading space of a continuation counts
        if len((current + ch).encode("utf-8")) > limit:
            out.append(current)
            current = ch
        else:
            current += ch
    out.append(current)
    return [out[0]] + [" " + part for part in out[1:]]


def ics_content(title: str, start: datetime, end: datetime, location: str | None, details: str | None,
                guests: list[str], now: datetime | None = None, uid: str | None = None) -> str:
    stamp = _utc_stamp(now or datetime.now(timezone.utc))
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Jarvis//Errands//EN", "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
        "BEGIN:VEVENT", f"UID:{uid or uuid.uuid4()}@jarvis", f"DTSTAMP:{stamp}",
        f"DTSTART:{_utc_stamp(start)}", f"DTEND:{_utc_stamp(end)}", f"SUMMARY:{_ics_text(_oneline(title))}",
    ]
    if location:
        lines.append(f"LOCATION:{_ics_text(_oneline(location))}")
    if details:
        lines.append(f"DESCRIPTION:{_ics_text(details)}")
    lines += [f"ATTENDEE;ROLE=REQ-PARTICIPANT:mailto:{g}" for g in guests]
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "".join(f"{part}\r\n" for line in lines for part in _fold(line))


def describe_span(start: datetime, end: datetime, tz: tzinfo | None = None) -> str:
    """'Thursday 12 November 2026, 15:00 to 15:30 (UTC+05:30)' for the user to check."""
    s, e = start.astimezone(tz), end.astimezone(tz)
    offset = s.strftime("%z")
    zone = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    name = s.tzname()
    label = f"{name}, {zone}" if name and not name.startswith("UTC") else zone
    if s.date() == e.date():
        return f"{s:%A %d %B %Y}, {s:%H:%M} to {e:%H:%M} ({label})"
    return f"{s:%A %d %B %Y %H:%M} to {e:%A %d %B %Y %H:%M} ({label})"


def resolve_times(start: str, end: str | None, now: datetime, tz: tzinfo | None = None) -> tuple[datetime, datetime]:
    """Parse the start (future, local) and the end (relative to the start). Aware datetimes."""
    from jarvis.proactive.schedule_math import localize

    naive_now = now.astimezone(tz).replace(tzinfo=None)
    try:
        start_naive = parse_when(start, naive_now)
    except WhenError as exc:
        raise ErrandError(f"cannot use the start time: {exc}") from exc
    s = localize(start_naive, tz)
    if end and end.strip():
        reference = s.astimezone(tz).replace(tzinfo=None)  # a start inside a DST gap has already moved
        try:
            e = localize(parse_when(end, reference), tz)
        except WhenError as exc:
            raise ErrandError(f"cannot use the end time: {exc}") from exc
    else:
        e = (s.astimezone(timezone.utc) + timedelta(minutes=DEFAULT_EVENT_MINUTES)).astimezone(tz)
    if e.astimezone(timezone.utc) <= s.astimezone(timezone.utc):
        raise ErrandError("the event must end after it starts")
    return s, e


# -- maps ----------------------------------------------------------------------------------------------
_TRAVEL = {"driving": "driving", "transit": "transit", "walking": "walking", "cycling": "bicycling"}


def map_url(query: str | None, origin: str | None, destination: str | None, mode: str) -> str:
    query, origin, destination = _oneline(query), _oneline(origin), _oneline(destination)
    if destination:
        q = _query([("api", "1"), ("origin", origin), ("destination", destination), ("travelmode", _TRAVEL.get(mode, "driving"))])
        return f"https://www.google.com/maps/dir/?{q}"
    if origin and not query:
        raise ErrandError("directions need a destination; give destination (and origin if it is not here)")
    if not query:
        raise ErrandError("give a place or search to look up (query), or a destination for directions")
    return f"https://www.google.com/maps/search/?{_query([('api', '1'), ('query', query)])}"


# -- messages -------------------------------------------------------------------------------------------
def phone_digits(to: str) -> str:
    """International digits only ('+91 98765-43210' -> '919876543210'). Raises ErrandError."""
    digits = re.sub(r"[\s\-().]", "", to.strip())
    if digits.startswith("00"):
        digits = digits[2:]
    digits = digits.lstrip("+")
    if not digits.isdigit() or not 7 <= len(digits) <= 15:
        raise ErrandError("give the phone number with its country code, for example +91 98765 43210")
    return digits


def message_url(app: str, to: str, text: str) -> tuple[str, str]:
    """(url, what the user must still do). Raises ErrandError."""
    if app == "whatsapp":
        return f"https://wa.me/{phone_digits(to)}?text={enc(text)}", "press send in WhatsApp"
    if app == "sms":
        number = "+" + phone_digits(to)
        return f"sms:{number}?body={enc(text)}", "press send in the messaging app"
    if app == "telegram":
        # Telegram has no public link that both picks a person and fills the text: this one
        # fills the text and lets the user choose who gets it.
        return f"https://t.me/share/url?url=%20&text={enc(text)}", f"choose {to.strip()[:60]} in Telegram, then press send"
    raise ErrandError(f"unknown messaging app '{app}'")


# -- providers --------------------------------------------------------------------------------------------
MAIL_PROVIDERS = ("gmail", "outlook", "mailto")
CALENDAR_PROVIDERS = ("google", "outlook", "ics")


def pick_provider(requested: str, preferred: str, allowed: tuple[str, ...], chrome: bool, web_default: str, fallback: str) -> str:
    """An explicit request wins, then the user's setting, then: the web default when Chrome
    is connected, else the app/file fallback."""
    for choice in (requested, preferred):
        choice = (choice or "auto").strip().lower()
        if choice in allowed:
            return choice
    return web_default if chrome else fallback


# -- opening ------------------------------------------------------------------------------------------------
async def chrome_connected(ctx: ToolContext) -> bool:
    """True when the Chrome extension is paired and connected right now."""
    try:
        from jarvis.tools.builtin import browser as b

        bridge = await b._bridge(ctx)
        return bool(bridge.connected)
    except Exception:
        log.debug("chrome not available", exc_info=True)
        return False


async def open_in_chrome(ctx: ToolContext, url: str) -> str:
    from jarvis.tools.builtin import browser as b

    reply = await b._send(ctx, "open", {"url": url, "new_tab": True}, timeout=45)
    tab = f" (tab_id={reply['tabId']})" if "tabId" in reply else ""
    return f"in the JARVIS tab group of Chrome{tab}"


def open_default(url: str) -> str:
    if not plat.current().open_url(url):
        raise OSError("no default application accepted the link")
    return "in the default application" if not url.lower().startswith(("http://", "https://")) else "in the default browser"


async def open_web(ctx: ToolContext, url: str, chrome: bool) -> str:
    if chrome:
        try:
            return await open_in_chrome(ctx, url)
        except Exception as exc:  # fall back rather than lose the errand
            log.warning("could not open in Chrome (%s); using the default browser", exc)
    return open_default(url)


def _fail(exc: Exception) -> ToolResult:
    return ToolResult.fail(str(exc), "Fix the arguments and call the tool again, or ask the user for what is missing.")


# -- the tools -------------------------------------------------------------------------------------------------
@tool(group="errands", timeout=75)
async def draft_email(
    to: str,
    subject: str,
    body: str,
    ctx: ToolContext,
    cc: str | None = None,
    bcc: str | None = None,
    provider: Literal["auto", "gmail", "outlook", "mailto"] = "auto",
) -> ToolResult:
    """Open a prefilled email compose window for the user to review. It NEVER sends.

    Gmail or Outlook on the web (in the user's signed-in Chrome when connected), or the
    default mail app via mailto. Tell the user the draft is open and they can read it and
    press Send themselves; if they ask you to send it, pressing Send through the browser
    tools asks them to confirm first.

    Args:
        to: Recipient address(es), separated by commas or semicolons.
        subject: Subject line.
        body: Message text; line breaks are kept.
        cc: Optional Cc address(es).
        bcc: Optional Bcc address(es).
        provider: auto uses the user's setting, else Gmail when Chrome is connected, else mailto.
    """
    try:
        recipients, ccs, bccs = addresses(to), addresses(cc, "cc"), addresses(bcc, "bcc")
        if not recipients:
            raise ErrandError("no recipient given; ask the user who the email is for")
        chrome = await chrome_connected(ctx)
        chosen = pick_provider(provider, ctx.settings.mail_provider, MAIL_PROVIDERS, chrome, "gmail", "mailto")
        url = email_url(chosen, recipients, subject, body, ccs, bccs)
        where = await open_web(ctx, url, chrome) if chosen != "mailto" else open_default(url)
    except (ErrandError, OSError) as exc:
        return _fail(exc)
    names = {"gmail": "Gmail", "outlook": "Outlook", "mailto": "the default mail app"}
    return ToolResult.success(
        f"Opened a draft in {names[chosen]} {where}: to {', '.join(recipients)}, subject \"{_oneline(subject)}\". "
        "It has NOT been sent; the user reviews it and sends it. If the page asks the user to sign in, "
        "tell them so and wait.",
        data={"provider": chosen, "url": url},
    )


@tool(group="errands", timeout=75)
async def create_calendar_event(
    title: str,
    start: str,
    ctx: ToolContext,
    end: str | None = None,
    location: str | None = None,
    details: str | None = None,
    guests: str | None = None,
    provider: Literal["auto", "google", "outlook", "ics"] = "auto",
) -> ToolResult:
    """Open a prefilled new-event page (Google or Outlook calendar), or save an .ics file and open it
    in the default calendar app. The event is NOT saved until the user presses Save.

    Read the echoed time back to the user so they can check it. If the page asks to sign in,
    tell the user and wait. Do not press Save unless the user asks (it asks them to confirm).

    Args:
        title: Event title.
        start: Start: ISO local datetime or "tomorrow 3pm", "friday 10am", "at 18:30".
        end: Optional end, measured from the start: "in 30 minutes", "4pm" or ISO. Default is one hour after the start.
        location: Optional place.
        details: Optional description.
        guests: Optional guest email address(es), separated by commas.
        provider: auto uses the user's setting, else Google when Chrome is connected, else an .ics file.
    """
    from jarvis.proactive import tasks as sched

    try:
        if not title.strip():
            raise ErrandError("the event needs a title")
        guest_list = addresses(guests, "guest")
        s, e = resolve_times(start, end, sched.local_now())
        chrome = await chrome_connected(ctx)
        chosen = pick_provider(provider, ctx.settings.calendar_provider, CALENDAR_PROVIDERS, chrome, "google", "ics")
        words = describe_span(s, e)
        if chosen == "ics":
            path = write_ics(title, s, e, location, details, guest_list)
            plat.current().open_path(str(path))
            where, url = f"as the file {path} (opened in the default calendar app)", str(path)
        else:
            url = calendar_url(chosen, title, s, e, location, details, guest_list)
            where = await open_web(ctx, url, chrome)
    except (ErrandError, OSError) as exc:
        return _fail(exc)
    return ToolResult.success(
        f"Opened a new event \"{_oneline(title)}\" ({chosen}) {where}: {words}. It is NOT saved yet; "
        "the user reviews it and presses Save. Read the time back to them. If the page asks to sign in, tell them and wait.",
        data={"provider": chosen, "url": url, "start": s.isoformat(), "end": e.isoformat()},
    )


def write_ics(title: str, start: datetime, end: datetime, location: str | None, details: str | None,
              guests: list[str], folder: Path | None = None) -> Path:
    folder = folder or known_folder("Documents")
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", _oneline(title)).strip("-")[:40] or "event"
    path = folder / f"jarvis-{slug}-{start.astimezone():%Y%m%d-%H%M}.ics"
    path.write_bytes(ics_content(title, start, end, location, details, guests).encode("utf-8"))
    return path


@tool(group="errands", timeout=75)
async def open_map(
    ctx: ToolContext,
    query: str | None = None,
    origin: str | None = None,
    destination: str | None = None,
    mode: Literal["driving", "transit", "walking", "cycling"] = "driving",
) -> ToolResult:
    """Open Google Maps: a place search, or directions between two places.

    Give query for "find a pharmacy near Connaught Place"; give destination (and origin
    when the start is not the user's current location) for directions. Open the page and
    read it with the browser tools to report ratings, distances or travel times.

    Args:
        query: What to search for or show on the map.
        origin: Where directions start from (empty: the user's current location).
        destination: Where directions go to.
        mode: driving, transit, walking or cycling (directions only).
    """
    try:
        url = map_url(query, origin, destination, mode)
        chrome = await chrome_connected(ctx)
        where = await open_web(ctx, url, chrome)
    except (ErrandError, OSError) as exc:
        return _fail(exc)
    what = f"directions to {_oneline(destination)} by {mode}" if destination else f"a search for {_oneline(query)}"
    return ToolResult.success(f"Opened Google Maps with {what} {where}.", data={"url": url})


@tool(group="errands", timeout=75)
async def compose_message(app: Literal["whatsapp", "telegram", "sms"], to: str, text: str, ctx: ToolContext) -> ToolResult:
    """Open a chat app with a message already typed, for the user to send. It NEVER sends.

    WhatsApp and SMS need the phone number with country code. Telegram opens its share page
    with the text filled in and the user picks the person. Tell the user what is left to do.

    Args:
        app: whatsapp, telegram or sms.
        to: Phone number with country code (for Telegram: the person's name or @username, shown to the user).
        text: The message text.
    """
    try:
        if not text.strip():
            raise ErrandError("the message is empty")
        url, todo = message_url(app, to, text)
        if urlparse(url).scheme in ("http", "https"):
            where = await open_web(ctx, url, await chrome_connected(ctx))
        else:
            where = open_default(url)
    except (ErrandError, OSError) as exc:
        return _fail(exc)
    return ToolResult.success(
        f"Opened {app} {where} with the message typed in. It has NOT been sent: the user must {todo}.",
        data={"url": url},
    )
