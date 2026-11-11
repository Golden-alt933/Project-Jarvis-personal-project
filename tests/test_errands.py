"""Errands: URL and .ics construction (escaping, time zones, providers) and the tools that
open them. No network and no real browser: Chrome and the OS are faked."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo

import pytest

from jarvis.config import Settings
from jarvis.proactive import tasks as sched
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import errands
from jarvis.tools.builtin.errands import (
    ErrandError,
    addresses,
    calendar_url,
    describe_span,
    email_url,
    ics_content,
    map_url,
    message_url,
    phone_digits,
    pick_provider,
    resolve_times,
)
from jarvis.tools.registry import ToolRegistry

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
KOL = ZoneInfo("Asia/Kolkata")


def qs(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query, keep_blank_values=True).items()}


# -- addresses -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a@x.com", ["a@x.com"]),
        ("a@x.com, b@y.org", ["a@x.com", "b@y.org"]),
        ("a@x.com;b@y.org ; c@z.io", ["a@x.com", "b@y.org", "c@z.io"]),
        ("Sam Lee <sam@example.com>", ["sam@example.com"]),
        ("first.last+tag@sub.example.co.uk", ["first.last+tag@sub.example.co.uk"]),
        (["a@x.com", "b@y.org, c@z.io"], ["a@x.com", "b@y.org", "c@z.io"]),
        (None, []),
        ("", []),
    ],
)
def test_addresses(text, expected):
    assert addresses(text) == expected


@pytest.mark.parametrize("bad", ["not-an-address", "a@b", "a@@b.com", "a b@c.com", "x@y.com\nBcc: evil@z.com extra", "<>", "@x.com"])
def test_addresses_reject(bad):
    with pytest.raises(ErrandError):
        addresses(bad)


# -- email URLs -----------------------------------------------------------------------------------------
SUBJECT = "Q&A: café, 50% off + more"
BODY = "Hi Zoë,\nI'll be 10 minutes late & bring 2+2 slides.\n\n日本語 ✓ #1 = 100%\nThanks"


@pytest.mark.parametrize("provider", ["gmail", "outlook", "mailto"])
def test_email_urls_round_trip_awkward_text(provider):
    url = email_url(provider, ["a@x.com", "b+tag@y.org"], SUBJECT, BODY, ["cc@x.com"], ["bcc@x.com"])
    if provider == "mailto":
        parsed = urlparse(url)
        assert parsed.scheme == "mailto"
        assert unquote(parsed.path) == "a@x.com,b+tag@y.org"
        got = qs(url)
        assert got["body"] == BODY.replace("\n", "\r\n")  # RFC 6068: CRLF
    else:
        got = qs(url)
        assert got["body"] == BODY
        assert got["to"] == "a@x.com,b+tag@y.org"
    assert got.get("su", got.get("subject")) == SUBJECT
    assert got["cc"] == "cc@x.com" and got["bcc"] == "bcc@x.com"


def test_email_url_shapes():
    gmail = email_url("gmail", ["a@x.com"], "Hi", "Line1\nLine2", [], [])
    assert gmail.startswith("https://mail.google.com/mail/?view=cm&fs=1&to=a@x.com&")
    assert "su=Hi" in gmail and "body=Line1%0ALine2" in gmail and "cc=" not in gmail
    outlook = email_url("outlook", ["a@x.com"], "Hi", "b", [], [])
    assert outlook.startswith("https://outlook.office.com/mail/deeplink/compose?to=a@x.com&subject=Hi&body=b")
    mailto = email_url("mailto", ["a@x.com"], "Hello there", "x y", [], [])
    assert mailto == "mailto:a@x.com?subject=Hello%20there&body=x%20y"  # %20, never '+'
    assert email_url("mailto", ["a@x.com"], "", "", [], []) == "mailto:a@x.com"


def test_plus_ampersand_and_hash_are_percent_encoded():
    url = email_url("gmail", ["a@x.com"], "1+1 & 2 # 3", "a+b&c#d", [], [])
    assert "1%2B1%20%26%202%20%23%203" in url and "a%2Bb%26c%23d" in url
    assert "+" not in url.split("to=a@x.com", 1)[1].replace("%2B", "")  # no raw plus signs besides the escaped ones


def test_email_subject_newlines_cannot_inject_headers():
    url = email_url("mailto", ["a@x.com"], "Hi\r\nBcc: evil@z.com", "b", [], [])
    assert "%0D%0A" not in url.split("subject=")[1].split("&")[0]
    assert qs(url)["subject"] == "Hi Bcc: evil@z.com"


def test_unknown_email_provider():
    with pytest.raises(ErrandError):
        email_url("yahoo", ["a@x.com"], "s", "b", [], [])


# -- calendar URLs -------------------------------------------------------------------------------------------
def span(start: datetime, minutes: int = 30):
    s = start.astimezone(UTC)
    return s, s + timedelta(minutes=minutes)


def test_google_calendar_url_carries_utc_times_whatever_the_local_zone():
    cases = [
        (datetime(2026, 11, 12, 15, 0, tzinfo=NY), "20261112T200000Z/20261112T203000Z"),  # EST, UTC-5
        (datetime(2026, 6, 12, 15, 0, tzinfo=NY), "20260612T190000Z/20260612T193000Z"),  # EDT, UTC-4
        (datetime(2026, 11, 12, 15, 0, tzinfo=KOL), "20261112T093000Z/20261112T100000Z"),  # UTC+5:30
        (datetime(2026, 3, 8, 15, 0, tzinfo=NY), "20260308T190000Z/20260308T193000Z"),  # the day the clocks change
    ]
    for start, dates in cases:
        s, e = span(start)
        url = calendar_url("google", "Design review", s, e, None, None, [])
        assert qs(url)["dates"] == dates, start


def test_google_calendar_url_fields_round_trip():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=NY))
    url = calendar_url("google", "R&D: café + plans", s, e, "Room 4/B & lobby", "Agenda:\n1. Intro\n2. Q&A", ["a@x.com", "b@y.org"])
    got = qs(url)
    assert got["action"] == "TEMPLATE"
    assert got["text"] == "R&D: café + plans" and got["location"] == "Room 4/B & lobby"
    assert got["details"] == "Agenda:\n1. Intro\n2. Q&A" and got["add"] == "a@x.com,b@y.org"
    assert url.startswith("https://calendar.google.com/calendar/render?action=TEMPLATE&text=")
    assert "location=" not in calendar_url("google", "x", s, e, None, None, [])


def test_outlook_calendar_url():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=KOL))
    url = calendar_url("outlook", "Sync", s, e, "HQ", "notes & more", ["a@x.com"])
    got = qs(url)
    assert got["startdt"] == "2026-11-12T09:30:00Z" and got["enddt"] == "2026-11-12T10:00:00Z"
    assert got["subject"] == "Sync" and got["body"] == "notes & more" and got["to"] == "a@x.com"
    assert got["path"] == "/calendar/action/compose" and got["rru"] == "addevent"
    assert url.startswith("https://outlook.office.com/calendar/0/deeplink/compose?")


def test_unknown_calendar_provider():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=NY))
    with pytest.raises(ErrandError):
        calendar_url("yahoo", "x", s, e, None, None, [])


# -- times ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("now", "start", "end", "start_utc", "end_utc"),
    [
        (datetime(2026, 11, 10, 14, 0, tzinfo=NY), "tomorrow 3pm", None, (2026, 11, 11, 20, 0), (2026, 11, 11, 21, 0)),  # default one hour
        (datetime(2026, 11, 10, 14, 0, tzinfo=NY), "tomorrow 3pm", "in 30 minutes", (2026, 11, 11, 20, 0), (2026, 11, 11, 20, 30)),
        (datetime(2026, 11, 10, 14, 0, tzinfo=NY), "tomorrow 3pm", "4pm", (2026, 11, 11, 20, 0), (2026, 11, 11, 21, 0)),
        (datetime(2026, 3, 7, 12, 0, tzinfo=NY), "tomorrow 3pm", "in 90 minutes", (2026, 3, 8, 19, 0), (2026, 3, 8, 20, 30)),  # EDT starts that day
        (datetime(2026, 10, 31, 12, 0, tzinfo=NY), "tomorrow 3pm", None, (2026, 11, 1, 20, 0), (2026, 11, 1, 21, 0)),  # EST starts that day
        (datetime(2026, 6, 10, 12, 0, tzinfo=KOL), "2026-06-12T18:30", "2026-06-12T19:15", (2026, 6, 12, 13, 0), (2026, 6, 12, 13, 45)),
        (datetime(2026, 3, 7, 12, 0, tzinfo=NY), "2026-03-08T02:30", None, (2026, 3, 8, 7, 30), (2026, 3, 8, 8, 30)),  # nonexistent wall time
    ],
)
def test_resolve_times(now, start, end, start_utc, end_utc):
    s, e = resolve_times(start, end, now, now.tzinfo)
    assert s.astimezone(UTC) == datetime(*start_utc, tzinfo=UTC)
    assert e.astimezone(UTC) == datetime(*end_utc, tzinfo=UTC)


@pytest.mark.parametrize(("start", "end"), [("whenever", None), ("2020-01-01T10:00", None), ("tomorrow 3pm", "garbage"), ("tomorrow 3pm", "2020-01-01T10:00")])
def test_resolve_times_errors(start, end):
    with pytest.raises(ErrandError):
        resolve_times(start, end, datetime(2026, 11, 10, 14, 0, tzinfo=NY), NY)


def test_describe_span_in_words():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=KOL))
    assert describe_span(s, e, KOL) == "Thursday 12 November 2026, 15:00 to 15:30 (IST, UTC+05:30)"
    s2, e2 = datetime(2026, 11, 12, 23, 30, tzinfo=NY), datetime(2026, 11, 13, 0, 30, tzinfo=NY)
    assert "Thursday 12 November 2026 23:30 to Friday 13 November 2026 00:30" in describe_span(s2, e2, NY)
    assert "EST" in describe_span(s2, e2, NY) and "UTC-05:00" in describe_span(s2, e2, NY)


# -- .ics ---------------------------------------------------------------------------------------------------------------------
def unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\r\n")[:-1]:
        if raw.startswith(" "):
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def test_ics_content_structure_and_utc_times():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=NY))
    text = ics_content("Design review", s, e, "Room 4", "Bring slides", ["a@x.com"],
                       now=datetime(2026, 11, 10, 9, 0, tzinfo=UTC), uid="fixed")
    assert text.endswith("\r\n") and "\n" not in text.replace("\r\n", "")
    lines = unfold(text)
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-1] == "END:VCALENDAR"
    assert "DTSTART:20261112T200000Z" in lines and "DTEND:20261112T203000Z" in lines and "DTSTAMP:20261110T090000Z" in lines
    assert "UID:fixed@jarvis" in lines and "SUMMARY:Design review" in lines and "LOCATION:Room 4" in lines
    assert "DESCRIPTION:Bring slides" in lines and "ATTENDEE;ROLE=REQ-PARTICIPANT:mailto:a@x.com" in lines
    assert lines.index("BEGIN:VEVENT") < lines.index("END:VEVENT")


def test_ics_escapes_special_characters_and_folds_long_lines_without_splitting_characters():
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=NY))
    details = "Agenda; items, one\\two\nsecond line " + "日本語 café " * 30
    text = ics_content("A, B; C", s, e, None, details, [], uid="u")
    raw_lines = text.split("\r\n")[:-1]
    assert all(len(line.encode("utf-8")) <= 75 for line in raw_lines)
    lines = unfold(text)
    assert "SUMMARY:A\\, B\\; C" in lines
    description = next(line for line in lines if line.startswith("DESCRIPTION:"))
    assert description.startswith("DESCRIPTION:Agenda\\; items\\, one\\\\two\\nsecond line ")
    assert description.endswith("日本語 café ")


def test_write_ics_names_the_file_and_stores_utf8(tmp_path: Path):
    s, e = span(datetime(2026, 11, 12, 15, 0, tzinfo=NY))
    path = errands.write_ics("Réunion / été?", s, e, None, None, [], folder=tmp_path / "Documents")
    assert path.parent == tmp_path / "Documents" and path.suffix == ".ics"
    assert path.name.startswith("jarvis-R-union-t-")
    assert "SUMMARY:Réunion / été?" in path.read_bytes().decode("utf-8")


# -- maps ------------------------------------------------------------------------------------------------------------------------
def test_map_search_and_directions():
    assert map_url("pharmacy near Connaught Place", None, None, "driving") == (
        "https://www.google.com/maps/search/?api=1&query=pharmacy%20near%20Connaught%20Place"
    )
    url = map_url(None, "India Gate, New Delhi", "Café Étoile & Bar", "transit")
    got = qs(url)
    assert got == {"api": "1", "origin": "India Gate, New Delhi", "destination": "Café Étoile & Bar", "travelmode": "transit"}
    assert qs(map_url(None, None, "Home", "cycling"))["travelmode"] == "bicycling"
    assert "origin" not in qs(map_url(None, None, "Home", "walking"))  # from the current location
    assert qs(map_url("x", None, "Y", "driving"))["destination"] == "Y"  # a destination means directions
    assert "dir" in map_url(None, None, "Y", "driving")


@pytest.mark.parametrize("args", [(None, None, None), ("  ", None, None), (None, "A", None)])
def test_map_errors(args):
    with pytest.raises(ErrandError):
        map_url(*args, "driving")


# -- messages ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "digits"),
    [("+91 98765-43210", "919876543210"), ("0091 98765 43210", "919876543210"), ("(415) 555-0100", "4155550100"), ("+44 20 7946 0958", "442079460958")],
)
def test_phone_digits(raw, digits):
    assert phone_digits(raw) == digits


@pytest.mark.parametrize("bad", ["", "call mum", "12345", "+1" + "2" * 20, "98765 4321O"])
def test_phone_digits_rejects(bad):
    with pytest.raises(ErrandError):
        phone_digits(bad)


def test_message_urls():
    text = "On my way & running 5+ min late\nsee you ✓"
    url, todo = message_url("whatsapp", "+91 98765-43210", text)
    assert url.startswith("https://wa.me/919876543210?text=") and qs(url)["text"] == text
    assert "send" in todo
    sms, _ = message_url("sms", "+91 98765 43210", text)
    assert sms.startswith("sms:+919876543210?body=") and unquote(sms.split("body=")[1]) == text
    tg, todo = message_url("telegram", "@priya", text)
    assert qs(tg)["text"] == text and "@priya" in todo
    with pytest.raises(ErrandError):
        message_url("signal", "+1 415 555 0100", "x")


# -- provider choice ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("requested", "preferred", "chrome", "expected"),
    [
        ("auto", "auto", True, "gmail"),
        ("auto", "auto", False, "mailto"),
        ("outlook", "auto", False, "outlook"),  # an explicit request wins, with or without Chrome
        ("mailto", "gmail", True, "mailto"),
        ("auto", "outlook", True, "outlook"),  # then the user's setting
        ("auto", "outlook", False, "outlook"),
        ("auto", "nonsense", True, "gmail"),
        ("", "", False, "mailto"),
        ("GMAIL", "auto", False, "gmail"),
    ],
)
def test_mail_provider_selection(requested, preferred, chrome, expected):
    assert pick_provider(requested, preferred, errands.MAIL_PROVIDERS, chrome, "gmail", "mailto") == expected


@pytest.mark.parametrize(
    ("requested", "preferred", "chrome", "expected"),
    [("auto", "auto", True, "google"), ("auto", "auto", False, "ics"), ("outlook", "auto", True, "outlook"), ("auto", "ics", True, "ics")],
)
def test_calendar_provider_selection(requested, preferred, chrome, expected):
    assert pick_provider(requested, preferred, errands.CALENDAR_PROVIDERS, chrome, "google", "ics") == expected


# -- the tools -------------------------------------------------------------------------------------------------------------------------
class Chrome:
    def __init__(self, connected: bool, fail: bool = False) -> None:
        self.connected, self.fail, self.opened = connected, fail, []

    async def is_connected(self, ctx) -> bool:
        return self.connected

    async def open(self, ctx, url: str) -> str:
        if self.fail:
            raise RuntimeError("extension disconnected")
        self.opened.append(url)
        return "in the JARVIS tab group of Chrome (tab_id=7)"

    page_text = "Ask Maps\nResults\nCaffè Tonino 4.3 (5,725)\nSorrento 4.5 (909)"

    async def read(self, ctx, url: str, wait_for, link_filter=None) -> dict:
        if self.fail:
            raise RuntimeError("extension disconnected")
        self.opened.append(url)
        return {"tabId": 7, "url": url, "title": "Google Maps", "text": self.page_text, "links": []}


@pytest.fixture
def rig(tmp_path: Path, fake_platform, monkeypatch):
    def build(connected: bool, fail: bool = False, **settings):
        chrome = Chrome(connected, fail)
        monkeypatch.setattr(errands, "chrome_connected", chrome.is_connected)
        monkeypatch.setattr(errands, "open_in_chrome", chrome.open)
        monkeypatch.setattr(errands, "read_in_chrome", chrome.read)
        ctx = ToolContext(settings=Settings(data_dir_override=str(tmp_path), **settings))
        return chrome, ctx

    monkeypatch.setattr(errands, "known_folder", lambda name: tmp_path / name)
    monkeypatch.setattr(sched, "local_now", lambda: datetime(2026, 11, 10, 14, 0, tzinfo=KOL))
    return build, fake_platform, tmp_path


async def test_draft_email_in_chrome_opens_gmail_and_never_sends(rig):
    build, plat, _ = rig
    chrome, ctx = build(True)
    res = await errands.draft_email("test@example.com", "Running late", "I'll be ten minutes late to the 3 pm meeting", ctx)
    assert res.ok
    assert chrome.opened and chrome.opened[0].startswith("https://mail.google.com/mail/?view=cm")
    assert qs(chrome.opened[0])["body"] == "I'll be ten minutes late to the 3 pm meeting"
    assert "NOT been sent" in res.content and "Gmail" in res.content and "tab_id=7" in res.content
    assert plat.urls == []


async def test_draft_email_without_chrome_uses_the_default_mail_app(rig):
    build, plat, _ = rig
    chrome, ctx = build(False)
    res = await errands.draft_email("a@x.com, b@y.org", "Hi", "Body", ctx, cc="c@z.io")
    assert res.ok and chrome.opened == []
    assert plat.urls and plat.urls[0].startswith("mailto:a@x.com,b@y.org?")
    assert "default mail app" in res.content and "NOT been sent" in res.content


async def test_draft_email_provider_setting_and_explicit_choice(rig):
    build, plat, _ = rig
    chrome, ctx = build(True, mail_provider="outlook")
    res = await errands.draft_email("a@x.com", "s", "b", ctx)
    assert chrome.opened[0].startswith("https://outlook.office.com/mail/deeplink/compose")
    res = await errands.draft_email("a@x.com", "s", "b", ctx, provider="mailto")
    assert plat.urls[-1].startswith("mailto:") and res.ok


async def test_draft_email_falls_back_when_chrome_drops_mid_errand(rig):
    build, plat, _ = rig
    chrome, ctx = build(True, fail=True)
    res = await errands.draft_email("a@x.com", "s", "b", ctx)
    assert res.ok and plat.urls and plat.urls[0].startswith("https://mail.google.com/")
    assert "default browser" in res.content


@pytest.mark.parametrize("to", ["", "nobody", "a@x.com, oops"])
async def test_draft_email_bad_recipient(rig, to):
    build, plat, _ = rig
    _, ctx = build(True)
    res = await errands.draft_email(to, "s", "b", ctx)
    assert not res.ok and plat.urls == []


async def test_calendar_event_in_chrome_opens_google_with_utc_and_echoes_the_time(rig):
    build, plat, _ = rig
    chrome, ctx = build(True)
    res = await errands.create_calendar_event("Design review", "tomorrow 3pm", ctx, end="in 30 minutes")
    assert res.ok, res.content
    got = qs(chrome.opened[0])
    start_utc = datetime(2026, 11, 11, 15, 0).astimezone(UTC)  # 15:00 in the machine's own zone
    stamp = lambda d: d.strftime("%Y%m%dT%H%M%SZ")  # noqa: E731
    assert got["text"] == "Design review"
    assert got["dates"] == f"{stamp(start_utc)}/{stamp(start_utc + timedelta(minutes=30))}"
    assert "Wednesday 11 November 2026, 15:00 to 15:30" in res.content and "NOT saved" in res.content


async def test_calendar_event_without_chrome_writes_an_ics_and_opens_it(rig):
    build, plat, tmp = rig
    chrome, ctx = build(False)
    res = await errands.create_calendar_event("Design review", "tomorrow 3pm", ctx, guests="a@x.com", location="Room 4")
    assert res.ok, res.content
    assert chrome.opened == [] and len(plat.paths) == 1
    path = Path(plat.paths[0])
    assert path.parent == tmp / "Documents" and path.suffix == ".ics" and path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "SUMMARY:Design review" in body and "LOCATION:Room 4" in body and "mailto:a@x.com" in body
    start_utc = datetime(2026, 11, 11, 15, 0).astimezone(UTC)
    assert f"DTSTART:{start_utc:%Y%m%dT%H%M%SZ}" in body
    assert f"DTEND:{start_utc + timedelta(hours=1):%Y%m%dT%H%M%SZ}" in body  # default length one hour


async def test_calendar_event_provider_setting_and_errors(rig):
    build, plat, _ = rig
    chrome, ctx = build(True, calendar_provider="outlook")
    assert (await errands.create_calendar_event("x", "tomorrow 3pm", ctx)).ok
    assert chrome.opened[0].startswith("https://outlook.office.com/calendar/")
    for args in (("", "tomorrow 3pm"), ("x", "whenever"), ("x", "2020-01-01T10:00")):
        res = await errands.create_calendar_event(*args, ctx)
        assert not res.ok


async def test_open_map_search_and_directions(rig):
    build, plat, _ = rig
    chrome, ctx = build(True)
    res = await errands.open_map(ctx, query="Italian restaurant near Connaught Place")
    assert res.ok and qs(chrome.opened[0])["query"] == "Italian restaurant near Connaught Place"
    res = await errands.open_map(ctx, origin="India Gate", destination="Café Étoile", mode="walking")
    assert "directions from India Gate to Café Étoile by walking" in res.content
    assert qs(chrome.opened[1])["travelmode"] == "walking"
    assert not (await errands.open_map(ctx)).ok
    _, ctx2 = build(False)
    await errands.open_map(ctx2, query="coffee")
    assert plat.urls and plat.urls[-1].startswith("https://www.google.com/maps/search/")


async def test_compose_message(rig):
    build, plat, _ = rig
    chrome, ctx = build(True)
    res = await errands.compose_message("whatsapp", "+91 98765 43210", "On my way", ctx)
    assert res.ok and chrome.opened[0] == "https://wa.me/919876543210?text=On%20my%20way"
    assert "NOT been sent" in res.content
    res = await errands.compose_message("sms", "+91 98765 43210", "Hi there", ctx)
    assert res.ok and plat.urls[-1] == "sms:+919876543210?body=Hi%20there"
    assert not (await errands.compose_message("whatsapp", "mum", "hi", ctx)).ok
    assert not (await errands.compose_message("whatsapp", "+91 98765 43210", "  ", ctx)).ok


async def test_os_refusal_is_a_clean_failure(rig):
    build, plat, _ = rig
    _, ctx = build(False)
    plat.open_error = OSError("no handler")
    res = await errands.draft_email("a@x.com", "s", "b", ctx)
    assert not res.ok and "no handler" in res.content


def test_errand_tools_are_registered_in_their_own_group_and_never_in_the_local_menu():
    registry = ToolRegistry().load_builtin()
    names = {"draft_email", "create_calendar_event", "open_map", "compose_message",
             "find_flights", "find_hotels", "find_trains"}
    assert {s.name for s in registry.specs(["errands"])} == names
    assert "errands" not in Settings().local_tool_groups
    for name in names:
        assert registry.get(name).risk.value == "safe"  # they only prefill; nothing leaves the machine
    assert registry.get("draft_email").parameters["properties"]["provider"]["enum"] == ["auto", "gmail", "outlook", "mailto"]


# -- travel searches -----------------------------------------------------------------------------------------------------------
def test_travel_links_carry_the_exact_dates():
    from datetime import date

    out = date(2026, 11, 13)
    one_way = qs(errands.flights_url("Delhi", "Mumbai", out, None, 1))
    assert one_way["q"] == "Flights from Delhi to Mumbai on 2026-11-13 one way"
    back = qs(errands.flights_url("Delhi", "Mumbai", out, date(2026, 11, 16), 2))
    assert back["q"] == "Flights from Delhi to Mumbai on 2026-11-13 returning 2026-11-16 for 2 adults"
    stay = qs(errands.hotels_url("Jaipur & around", date(2026, 11, 14), date(2026, 11, 15), 2, 1, "rating"))
    assert stay == {"ss": "Jaipur & around", "checkin": "2026-11-14", "checkout": "2026-11-15",
                    "group_adults": "2", "no_rooms": "1", "order": "bayesian_review_score"}
    assert qs(errands.trains_url("Delhi", "Jaipur", date(2026, 11, 5)))["q"] == "trains from Delhi to Jaipur on 5 November 2026"


def test_trim_results_drops_the_filters_and_caps_the_length():
    page = "Skip to main content\nFilters   Pool  12\n\nJaipur: 1,314 properties found\nHome of the world  9.8\n" + "x" * 9000
    text = errands.trim_results(page, "properties found", limit=200)
    assert text.startswith("Jaipur: 1,314 properties found\nHome of the world 9.8")
    assert "Filters" not in text and "more results on the page" in text and len(text) < 300
    assert errands.trim_results("no marker here", "properties found") == "no marker here"


class Reader:
    def __init__(self, text: str = "53 results returned.\n8:45 AM IndiGo Nonstop ₹6,301", fail: bool = False) -> None:
        self.text, self.fail, self.calls = text, fail, []

    links: list = []

    async def __call__(self, ctx, url: str, wait_for, link_filter=None):
        self.calls.append((url, wait_for))
        if self.fail:
            raise RuntimeError("tab closed")
        return {"tabId": 7, "url": url, "title": "Results", "text": "Skip to main content\n" + self.text,
                "links": self.links if link_filter else []}


async def test_find_flights_resolves_the_day_and_returns_the_results(rig, monkeypatch):
    build, plat, _ = rig
    _, ctx = build(True)
    reader = Reader()
    monkeypatch.setattr(errands, "read_in_chrome", reader)
    res = await errands.find_flights("Delhi", "Mumbai", "next friday", ctx)  # the rig's today is Tue 10 Nov 2026
    assert res.ok and "Friday 13 November 2026, one way" in res.content
    assert qs(reader.calls[0][0])["q"] == "Flights from Delhi to Mumbai on 2026-11-13 one way"
    assert "IndiGo Nonstop ₹6,301" in res.content and "Skip to main content" not in res.content
    assert "<page_content" in res.content and res.data["read"] is True and res.data["tabId"] == 7
    res = await errands.find_flights("Delhi", "Mumbai", "13 nov", ctx, return_date="16 nov", adults=2)
    assert "returning Monday 16 November 2026" in res.content


async def test_find_flights_refuses_bad_input(rig, monkeypatch):
    build, _, _ = rig
    _, ctx = build(True)
    monkeypatch.setattr(errands, "read_in_chrome", Reader())
    for args, kw in ((("Delhi", "Mumbai", "whenever"), {}), (("", "Mumbai", "tomorrow"), {}),
                     (("Delhi", "Mumbai", "yesterday"), {}), (("Delhi", "Mumbai", "15 nov"), {"return_date": "13 nov"}),
                     (("Delhi", "Mumbai", "tomorrow"), {"adults": 0})):
        res = await errands.find_flights(*args, ctx, **kw)
        assert not res.ok, args


async def test_find_hotels_and_trains(rig, monkeypatch):
    build, _, _ = rig
    _, ctx = build(True)
    reader = Reader("Jaipur: 1,314 properties found\nHome of the world Scored 9.8 ₹ 4,400")
    monkeypatch.setattr(errands, "read_in_chrome", reader)
    res = await errands.find_hotels("Jaipur", "this saturday", ctx, nights=2, sort="price")
    q = qs(reader.calls[0][0])
    assert res.ok and (q["checkin"], q["checkout"], q["order"]) == ("2026-11-14", "2026-11-16", "price")
    assert "Saturday 14 November 2026 for 2 night(s)" in res.content and "Scored 9.8" in res.content
    assert not (await errands.find_hotels("Jaipur", "tomorrow", ctx, nights=0)).ok
    res = await errands.find_trains("Delhi", "Jaipur", "tomorrow", ctx)
    assert res.ok and "Wednesday 11 November 2026" in res.content
    assert qs(reader.calls[-1][0])["q"] == "trains from Delhi to Jaipur on 11 November 2026"


async def test_travel_search_without_chrome_opens_the_default_browser_and_says_so(rig, monkeypatch):
    build, plat, _ = rig
    _, ctx = build(False)
    res = await errands.find_flights("Delhi", "Mumbai", "tomorrow", ctx)
    assert res.ok and plat.urls[-1].startswith("https://www.google.com/travel/flights?")
    assert "could not be read" in res.content and res.data["read"] is False
    _, ctx = build(True)
    monkeypatch.setattr(errands, "read_in_chrome", Reader(fail=True))
    res = await errands.find_hotels("Jaipur", "tomorrow", ctx)
    assert res.ok and "tab closed" in res.content and plat.urls[-1].startswith("https://www.booking.com/")


def test_property_links_are_one_clean_dated_address_per_property():
    items = [
        {"label": "", "href": "https://www.booking.com/hotel/in/pathik-niwas.en-gb.html?label=gen173&sid=abc#hotelTmpl"},
        {"label": "Pathik Niwas  Opens in new window", "href": "https://www.booking.com/hotel/in/pathik-niwas.en-gb.html?label=gen173&sid=abc"},
        {"label": "See availability", "href": "https://www.booking.com/hotel/in/pathik-niwas.en-gb.html?label=gen173"},
        {"label": "Cocoon Auberge Opens in new window", "href": "https://www.booking.com/hotel/in/cocoon.html?x=1"},
        {"label": "Sign in", "href": "https://account.booking.com/sign-in"},
    ]
    keep = {"checkin": "2026-11-14", "checkout": "2026-11-15", "group_adults": "2", "no_rooms": "1"}
    assert errands.property_links(items, "/hotel/", keep) == [
        "- Pathik Niwas: https://www.booking.com/hotel/in/pathik-niwas.en-gb.html?checkin=2026-11-14&checkout=2026-11-15&group_adults=2&no_rooms=1",
        "- Cocoon Auberge: https://www.booking.com/hotel/in/cocoon.html?checkin=2026-11-14&checkout=2026-11-15&group_adults=2&no_rooms=1",
    ]
    assert len(errands.property_links(items, "/hotel/", keep, most=1)) == 1


async def test_find_hotels_lists_a_page_for_each_property(rig, monkeypatch):
    build, _, _ = rig
    _, ctx = build(True)
    reader = Reader("Jaipur: 2 properties found\nPathik Niwas Scored 9.6 ₹ 2,570")
    reader.links = [{"label": "Pathik Niwas Opens in new window", "href": "https://www.booking.com/hotel/in/pathik-niwas.html?sid=1"}]
    monkeypatch.setattr(errands, "read_in_chrome", reader)
    res = await errands.find_hotels("Jaipur", "this saturday", ctx)
    assert "- Pathik Niwas: https://www.booking.com/hotel/in/pathik-niwas.html?checkin=2026-11-14&checkout=2026-11-15" in res.content
    res = await errands.find_flights("Delhi", "Mumbai", "tomorrow", ctx)
    assert "Page of each property" not in res.content


async def test_calendar_event_length_in_minutes(rig):
    build, _, _ = rig
    chrome, ctx = build(True)
    res = await errands.create_calendar_event("Design review", "tomorrow 3pm", ctx, minutes=30)
    assert res.ok and "15:00 to 15:30" in res.content
    res = await errands.create_calendar_event("Design review", "tomorrow 3pm", ctx, end="5pm", minutes=30)
    assert "15:00 to 17:00" in res.content  # an explicit end wins
    assert not (await errands.create_calendar_event("x", "tomorrow 3pm", ctx, minutes=0)).ok


async def test_open_map_returns_what_the_map_shows(rig):
    build, plat, _ = rig
    chrome, ctx = build(True)
    res = await errands.open_map(ctx, query="Italian restaurant near Connaught Place")
    assert "Sorrento 4.5 (909)" in res.content and "Ask Maps" not in res.content  # trimmed to the results
    chrome.page_text = "Best\n12 min\n3.4 km\nvia Kasturba Gandhi Marg"
    res = await errands.open_map(ctx, origin="India Gate", destination="Sorrento, Ashoka Road")
    assert "directions from India Gate to Sorrento, Ashoka Road by driving" in res.content
    assert "12 min" in res.content and "3.4 km" in res.content
    res = await errands.open_map(ctx, destination="India Gate")
    assert "from the user's current location to India Gate" in res.content
    chrome.fail = True  # Chrome drops out: the default browser still gets the link, and nothing is described
    res = await errands.open_map(ctx, query="coffee")
    assert res.ok and plat.urls and "could not be read" in res.content
