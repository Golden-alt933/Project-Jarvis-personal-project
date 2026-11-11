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
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.parse import urlparse

from jarvis import platform as plat
from jarvis.tools.base import ToolContext, ToolResult, tool
from jarvis.tools.builtin.productivity import WhenError, parse_day, parse_when
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
    return "(default app)" if not url.lower().startswith(("http://", "https://")) else "in the default browser"


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
        f"Opened a draft in {names[chosen]}{'' if where == '(default app)' else ' ' + where}: to {', '.join(recipients)}, subject \"{_oneline(subject)}\". "
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
    minutes: int | None = None,
) -> ToolResult:
    """Open a prefilled new-event page (Google or Outlook calendar), or save an .ics file and open it
    in the default calendar app. The event is NOT saved until the user presses Save.

    Read the echoed time back to the user so they can check it. If the page asks to sign in,
    tell the user and wait. Do not press Save unless the user asks (it asks them to confirm).

    Args:
        title: Event title.
        start: Start: ISO local datetime or "tomorrow 3pm", "friday 10am", "at 18:30".
        end: Optional end time: "4pm" or ISO. Leave empty when you give minutes.
        minutes: How long the event lasts, in minutes (a "30-minute event" is minutes=30). Default is 60.
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
        if minutes is not None and not (end and end.strip()):
            if not 1 <= int(minutes) <= 7 * 24 * 60:
                raise ErrandError("minutes must be between 1 and 10080 (a week)")
            end = f"in {int(minutes)} minutes"
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

    Give query alone for "find a pharmacy near Connaught Place": the result lists the places
    found with their ratings. Give origin and destination for directions: the result gives
    the travel time and distance. For "find a place and show directions from X" call it
    twice: first the search, then directions with origin=X and destination=the place chosen.

    Args:
        query: What to search for (a place search; leave empty for directions).
        origin: Where the trip starts, e.g. "India Gate" in "directions from India Gate" (empty: the user's current location).
        destination: Where the trip ends.
        mode: driving, transit, walking or cycling (directions only).
    """
    from jarvis.tools.builtin import browser as b

    try:
        url = map_url(query, origin, destination, mode)
        page = None
        if await chrome_connected(ctx):
            try:
                page = await read_in_chrome(ctx, url, " min" if destination else "Results")
            except Exception as exc:  # fall back rather than lose the errand
                log.warning("could not open the map in Chrome (%s); using the default browser", exc)
        if page is None:
            where = open_default(url)
        else:
            tab = f" (tab_id={page['tabId']})" if page.get("tabId") is not None else ""
            where = f"in the JARVIS tab group of Chrome{tab}"
    except (ErrandError, OSError) as exc:
        return _fail(exc)
    start = f" from {_oneline(origin)}" if origin and origin.strip() else " from the user's current location"
    what = (f"directions{start} to {_oneline(destination)} by {mode}" if destination
            else f"a search for {_oneline(query)}")
    message = f"Opened Google Maps with {what} {where}."
    if page is not None:
        shown = trim_results(page["text"], None if destination else "Results", limit=3000) or "(the map showed no text yet)"
        message += ("\nWhat the map shows:\n" + b.page_block(page["url"], page["title"], shown)
                    + "\nReport only places, ratings, times and distances listed here.")
    else:
        message += " Its contents could not be read, so do not describe them."
    return ToolResult.success(message, data={"url": url})


# -- travel searches -------------------------------------------------------------------------------------
# Dated searches are where a model goes wrong on its own: it has to work out "next Friday",
# find a site that shows prices without signing in, and fill a date picker. These tools do
# that part deterministically (the date is resolved here, the link carries it) and hand back
# the live results as page text, in a tab the browser tools can carry on from.
TRAVEL_RESULT_CHARS = 6000
HOTEL_ORDER = {"rating": "bayesian_review_score", "price": "price", "popular": "popularity"}


def _local_now() -> datetime:
    from jarvis.proactive import tasks as sched  # imported late: the scheduler imports the tools

    return sched.local_now()


def travel_day(text: str, now: datetime) -> date:
    try:
        return parse_day(text, now)
    except WhenError as exc:
        raise ErrandError(f"cannot use the date '{_oneline(text)}': {exc}") from exc


def _place(text: str | None, label: str) -> str:
    place = _oneline(text)
    if not place:
        raise ErrandError(f"the {label} is missing")
    return place


def _count(value: int, label: str, most: int) -> int:
    if not 1 <= int(value) <= most:
        raise ErrandError(f"{label} must be between 1 and {most}")
    return int(value)


def _spoken(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B %Y}"


def flights_url(origin: str, destination: str, depart: date, back: date | None, adults: int) -> str:
    words = f"Flights from {origin} to {destination} on {depart.isoformat()}"
    words += f" returning {back.isoformat()}" if back else " one way"
    if adults > 1:
        words += f" for {adults} adults"
    return "https://www.google.com/travel/flights?" + _query([("q", words), ("hl", "en")])


def hotels_url(place: str, checkin: date, checkout: date, adults: int, rooms: int, sort: str) -> str:
    return "https://www.booking.com/searchresults.html?" + _query([
        ("ss", place), ("checkin", checkin.isoformat()), ("checkout", checkout.isoformat()),
        ("group_adults", str(adults)), ("no_rooms", str(rooms)), ("order", HOTEL_ORDER[sort]),
    ])


def trains_url(origin: str, destination: str, day: date) -> str:
    words = f"trains from {origin} to {destination} on {day.day} {day:%B %Y}"
    return "https://www.google.com/search?" + _query([("q", words), ("hl", "en")])


def trim_results(text: str, marker: str | None, limit: int = TRAVEL_RESULT_CHARS) -> str:
    """Drop the navigation and filter lists before the results, and cap the length."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if marker and (at := text.find(marker)) > 0:
        text = text[text.rfind("\n", 0, at) + 1:]
    if len(text) > limit:
        text = text[:limit] + "\n[... more results on the page: browser_read shows the rest]"
    return text


def property_links(items: list[dict[str, Any]], path_part: str, keep: dict[str, str], most: int = 15) -> list[str]:
    """'Name: address' lines for the result links on a page, one per property.

    A results page links each property several times (photo, title, button) with long
    tracking addresses. Keep the first named link per property and rebuild its address from
    the path plus the search's own parameters, so opening it shows the same dates.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        href = str(item.get("href") or "")
        parsed = urlparse(href)
        if path_part not in parsed.path or parsed.path in seen:
            continue
        name = re.sub(r"\s*Opens in new window\s*$", "", " ".join(str(item.get("label") or "").split()), flags=re.I)
        if not name or name.lower() in ("see availability", "show prices"):
            continue
        seen.add(parsed.path)
        lines.append(f"- {name[:80]}: {parsed.scheme}://{parsed.netloc}{parsed.path}?{_query(list(keep.items()))}")
        if len(lines) >= most:
            break
    return lines


async def read_in_chrome(ctx: ToolContext, url: str, wait_for: str | None, link_filter: str | None = None) -> dict[str, Any]:
    """Open url in the JARVIS tab group, wait for the results, and return the page text
    (and, with link_filter, the page's links whose address contains it)."""
    from jarvis.tools.builtin import browser as b

    opened = await b._send(ctx, "open", {"url": url, "new_tab": True}, timeout=45)
    if wait_for:
        try:
            await b._send(ctx, "wait", {"text": wait_for, "timeout_ms": 12000}, timeout=20)
        except Exception:  # the page may word it differently: read whatever is there
            log.debug("results marker %r did not appear", wait_for, exc_info=True)
    await b._sleep(ctx, 2.0)  # prices keep arriving for a moment after the list appears
    page = await b._send(ctx, "read", {"max_chars": 20000})
    links: list[dict[str, Any]] = []
    if link_filter:
        try:
            for offset in (0, 80):  # a results page lists each property two or three times
                found = await b._send(ctx, "inspect", {"filter": link_filter, "offset": offset})
                links += list(found.get("items") or [])
                if found.get("next_offset") is None:
                    break
        except Exception:  # the results text is still worth returning
            log.debug("could not list result links", exc_info=True)
    return {"tabId": opened.get("tabId"), "url": page.get("url", url), "title": page.get("title", ""),
            "text": str(page.get("text", "")), "links": links}


async def _travel_search(ctx: ToolContext, url: str, heading: str, site: str, *, wait_for: str | None,
                         marker: str | None, link_part: str | None = None,
                         link_keep: dict[str, str] | None = None) -> ToolResult:
    from jarvis.tools.builtin import browser as b

    if not await chrome_connected(ctx):
        where = open_default(url)
        return ToolResult.success(
            f"{heading}\nOpened the {site} results {where}. Chrome is not connected to Jarvis, so the "
            "results could not be read: tell the user they are on screen, or that connecting the "
            "Jarvis Chrome extension lets you read and compare them.",
            data={"url": url, "read": False},
        )
    try:
        page = await read_in_chrome(ctx, url, wait_for, link_part)
    except Exception as exc:
        log.warning("could not read %s results in Chrome (%s); opening the default browser", site, exc)
        where = open_default(url)
        return ToolResult.success(
            f"{heading}\nOpened the {site} results {where}, but they could not be read ({exc}).",
            data={"url": url, "read": False},
        )
    body = trim_results(page["text"], marker) or "(the page showed no text)"
    tab = f" (tab_id={page['tabId']})" if page.get("tabId") is not None else ""
    links = property_links(page.get("links") or [], link_part, link_keep or {}) if link_part else []
    if links:
        body += ("\n\nPage of each property, for these dates (pass one to browser_open with new_tab=false "
                 "to see its rooms and its reserve button):\n" + "\n".join(links))
    return ToolResult.success(
        f"{heading}\nLive results from {site}, open in Chrome{tab}:\n"
        f"{b.page_block(page['url'], page['title'], body)}\n"
        "Report the options that fit (price, times, name) and say which date you searched. To go "
        "further on this tab use browser_inspect and browser_click; the user signs in and pays themselves.",
        data={"url": url, "read": True, "tabId": page.get("tabId")},
    )


@tool(group="errands", timeout=90)
async def find_flights(
    origin: str,
    destination: str,
    date: str,
    ctx: ToolContext,
    return_date: str | None = None,
    adults: int = 1,
) -> ToolResult:
    """Look up live flight prices and times for a date and return the list of flights.

    Use this for any "find/compare/book flights" request instead of web_search: it searches
    Google Flights for the exact date and returns airlines, times, stops and prices. It only
    searches; nothing is booked.

    Args:
        origin: City or airport to fly from, e.g. "Delhi" or "DEL".
        destination: City or airport to fly to.
        date: Departure day, as the user said it or as YYYY-MM-DD: "next Friday", "13 Nov", "tomorrow".
        return_date: Return day for a round trip, same forms (empty: one way).
        adults: Number of adult travellers.
    """
    try:
        now = _local_now()
        start, end = _place(origin, "origin"), _place(destination, "destination")
        depart = travel_day(date, now)
        back = travel_day(return_date, now) if return_date and return_date.strip() else None
        if back is not None and back < depart:
            raise ErrandError("the return date is before the departure date")
        url = flights_url(start, end, depart, back, _count(adults, "adults", 9))
        trip = f"returning {_spoken(back)}" if back else "one way"
        heading = f"Flights from {start} to {end} on {_spoken(depart)}, {trip}, {adults} adult(s)."
        return await _travel_search(ctx, url, heading, "Google Flights", wait_for="results returned",
                                    marker="results returned")
    except (ErrandError, OSError) as exc:
        return _fail(exc)


@tool(group="errands", timeout=90)
async def find_hotels(
    place: str,
    checkin: str,
    ctx: ToolContext,
    nights: int = 1,
    adults: int = 2,
    rooms: int = 1,
    sort: Literal["rating", "price", "popular"] = "rating",
) -> ToolResult:
    """Look up hotels and other places to stay overnight on given dates, with prices and review scores.

    Not for restaurants, cafes or other places to visit: use open_map for those.
    Use this for any "find/compare/book a hotel" request instead of web_search: it searches
    Booking.com for the exact dates and returns names, areas, review scores and prices per
    stay. It only searches; nothing is booked.

    Args:
        place: City, area or landmark to stay near, e.g. "Jaipur".
        checkin: Check-in day, as the user said it or as YYYY-MM-DD: "this Saturday", "14 Nov".
        nights: Number of nights.
        adults: Number of adults.
        rooms: Number of rooms.
        sort: "rating" for the best reviewed first, "price" for the cheapest first, "popular" for the site's default.
    """
    try:
        where = _place(place, "place")
        first = travel_day(checkin, _local_now())
        last = first + timedelta(days=_count(nights, "nights", 30))
        url = hotels_url(where, first, last, _count(adults, "adults", 30), _count(rooms, "rooms", 30), sort)
        heading = (f"Stays in {where} from {_spoken(first)} for {nights} night(s), {adults} adult(s), "
                   f"{rooms} room(s), sorted by {sort}. Prices are for the whole stay.")
        keep = {"checkin": first.isoformat(), "checkout": last.isoformat(),
                "group_adults": str(adults), "no_rooms": str(rooms)}
        return await _travel_search(ctx, url, heading, "Booking.com", wait_for="properties found",
                                    marker="properties found", link_part="/hotel/", link_keep=keep)
    except (ErrandError, OSError) as exc:
        return _fail(exc)


@tool(group="errands", timeout=90)
async def find_trains(origin: str, destination: str, date: str, ctx: ToolContext) -> ToolResult:
    """Look up trains between two places on a date: names, numbers, departure and arrival times.

    Use this for any "find/compare/book a train" request instead of web_search. It returns a
    timetable with typical fares; seat availability and booking happen on the rail site.

    Args:
        origin: City or station to travel from.
        destination: City or station to travel to.
        date: Travel day, as the user said it or as YYYY-MM-DD: "tomorrow", "next Friday", "13 Nov".
    """
    try:
        start, end = _place(origin, "origin"), _place(destination, "destination")
        day = travel_day(date, _local_now())
        heading = f"Trains from {start} to {end} on {_spoken(day)}."
        return await _travel_search(ctx, trains_url(start, end, day), heading, "Google Search",
                                    wait_for=None, marker="Search Results")
    except (ErrandError, OSError) as exc:
        return _fail(exc)


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
        f"Opened {app} {'' if where == '(default app)' else where + ' '}with the message typed in. It has NOT been sent: the user must {todo}.",
        data={"url": url},
    )
