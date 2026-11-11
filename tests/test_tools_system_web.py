"""Tests for the system, web and productivity tools. Offline; no machine state is changed."""

from __future__ import annotations

import socket
import sqlite3
import subprocess
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from jarvis.platform import PlatformUnsupported
from jarvis.tools.base import Risk, ToolContext
from jarvis.tools.builtin import productivity as prod
from jarvis.tools.builtin import system, web


# ---------------------------------------------------------------- fixtures
class FakeMemory:
    """Minimal stand-in for MemoryStore's generic access API."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def ensure_schema(self, ddl: str) -> None:
        self.conn.execute(ddl)
        self.conn.commit()

    def execute(self, sql, params=()):
        cur = self.conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        self.conn.commit()
        return rows


def _real_store(tmp_path):
    from jarvis.memory.store import MemoryStore  # may not exist yet

    return MemoryStore(tmp_path / "t.db")


def _memory_params():
    return ["fake", "real"]


@pytest.fixture(params=_memory_params())
def ctx(request, tmp_path):
    if request.param == "fake":
        memory = FakeMemory()
    else:
        try:
            memory = _real_store(tmp_path)
        except Exception as exc:  # store not written yet or incompatible
            pytest.skip(f"real MemoryStore unavailable: {exc}")
    return ToolContext(settings=SimpleNamespace(), memory=memory)


FAKE_DNS = {
    "example.com": "93.184.216.34",
    "localhost": "127.0.0.1",
    "internal.corp.example": "10.1.2.3",
    "api.open-meteo.com": "104.20.1.1",
    "geocoding-api.open-meteo.com": "104.20.1.2",
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    def getaddrinfo(host, port, *a, **k):
        if host in FAKE_DNS:
            ip = FAKE_DNS[host]
        else:
            try:
                socket.inet_pton(socket.AF_INET6 if ":" in host else socket.AF_INET, host)
                ip = host
            except OSError:
                raise socket.gaierror("unknown host")
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(web.socket, "getaddrinfo", getaddrinfo)


def _mock_web(monkeypatch, handler):
    monkeypatch.setattr(web, "_TRANSPORT", httpx.MockTransport(handler))


# ---------------------------------------------------------------- fetch_url
HTML = """<html><head><title>My Page</title><style>.x{}</style></head><body>
<nav>MENU LINKS</nav><header>SITE HEADER</header>
<article><h1>Headline</h1><p>First paragraph.</p><script>evil()</script><p>Second.</p></article>
<footer>COPYRIGHT</footer></body></html>"""


def test_extract_text_prefers_article_and_drops_chrome():
    title, text = web.extract_text(HTML)
    assert title == "My Page"
    assert "Headline" in text and "First paragraph." in text and "Second." in text
    for junk in ("MENU LINKS", "COPYRIGHT", "evil()", "SITE HEADER"):
        assert junk not in text


def test_fetch_url_html(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(200, html=HTML))
    r = web.fetch_url("https://example.com/a")
    assert r.ok and "Title: My Page" in r.content and "First paragraph." in r.content


def test_fetch_url_sends_browser_user_agent(monkeypatch):
    seen = {}

    def handler(req):
        seen["ua"] = req.headers["user-agent"]
        return httpx.Response(200, html=HTML)

    _mock_web(monkeypatch, handler)
    web.fetch_url("https://example.com/")
    assert "Mozilla/5.0" in seen["ua"]


def test_fetch_url_truncates(monkeypatch):
    big = "<html><body><main>" + "<p>word</p>" * 5000 + "</main></body></html>"
    _mock_web(monkeypatch, lambda req: httpx.Response(200, html=big))
    r = web.fetch_url("https://example.com/", max_chars=1000)
    assert r.ok and "truncated" in r.content and len(r.content) < 1400


def test_fetch_url_json_and_plain(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(200, json={"a": 1}))
    assert '"a"' in web.fetch_url("https://example.com/x.json").content
    _mock_web(monkeypatch, lambda req: httpx.Response(200, text="hello", headers={"content-type": "text/plain"}))
    assert "hello" in web.fetch_url("https://example.com/x.txt").content


def test_fetch_url_refuses_binary(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}))
    r = web.fetch_url("https://example.com/x.pdf")
    assert not r.ok and "application/pdf" in r.content


def test_fetch_url_http_error(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(404, text="nope"))
    r = web.fetch_url("https://example.com/missing")
    assert not r.ok and "404" in r.content


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/admin",
        "http://10.0.0.5/",
        "http://192.168.1.1/router",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost/",
        "http://localhost:11434/api/tags",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "http://internal.corp.example/",  # hostname resolving to a private IP
    ],
)
def test_fetch_url_refuses_private(monkeypatch, url):
    def handler(req):
        raise AssertionError("request must not be sent")

    _mock_web(monkeypatch, handler)
    r = web.fetch_url(url)
    assert not r.ok and ("private" in r.content or "local" in r.content)


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "ftp://example.com/x", "javascript:alert(1)", "example.com"])
def test_fetch_url_refuses_non_http(url):
    r = web.fetch_url(url)
    assert not r.ok and "http" in r.content


def test_fetch_url_blocks_redirect_to_private(monkeypatch):
    def handler(req):
        if req.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://192.168.0.1/secret"})
        raise AssertionError("private host must not be contacted")

    _mock_web(monkeypatch, handler)
    r = web.fetch_url("https://example.com/")
    assert not r.ok and "private" in r.content


def test_fetch_url_follows_safe_redirect(monkeypatch):
    def handler(req):
        if req.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, html=HTML)

    _mock_web(monkeypatch, handler)
    assert web.fetch_url("https://example.com/old").ok


def test_fetch_url_timeout(monkeypatch):
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    _mock_web(monkeypatch, handler)
    r = web.fetch_url("https://example.com/")
    assert not r.ok and "timed out" in r.content


# ---------------------------------------------------------------- web_search
class FakeDDGS:
    results: list = []
    error: Exception | None = None
    last: dict = {}

    def text(self, query, **kw):
        FakeDDGS.last = {"query": query, **kw}
        if FakeDDGS.error:
            raise FakeDDGS.error
        return FakeDDGS.results


@pytest.fixture
def ddgs(monkeypatch):
    FakeDDGS.results, FakeDDGS.error = [], None
    monkeypatch.setattr(web, "DDGS", FakeDDGS)
    return FakeDDGS


def test_web_search_formats_results(ddgs):
    ddgs.results = [
        {"title": "Python 3.13", "href": "https://python.org/3.13", "body": "What's new"},
        {"title": "no url", "href": "", "body": "dropped"},
    ]
    r = web.web_search("python 3.13", max_results=3, recency="week")
    assert r.ok and "https://python.org/3.13" in r.content and "dropped" not in r.content
    assert ddgs.last["timelimit"] == "w" and ddgs.last["max_results"] == 5  # two spare for dropped ads
    assert r.data[0]["title"] == "Python 3.13"


def test_web_search_no_results(ddgs):
    r = web.web_search("zzzz", recency="day")
    assert not r.ok and "no results" in r.content and "Hint" in r.content


def test_web_search_rate_limit(ddgs):
    ddgs.error = web.RatelimitException("429")
    r = web.web_search("x")
    assert not r.ok and "rate limit" in r.content and "Hint" in r.content


def test_web_search_other_error(ddgs):
    ddgs.error = RuntimeError("boom")
    r = web.web_search("x")
    assert not r.ok and "boom" in r.content


# ---------------------------------------------------------------- weather
GEO = {"results": [{"name": "Delhi", "admin1": "Delhi", "country": "India", "latitude": 28.65, "longitude": 77.23}]}
FORECAST = {
    "current_units": {"temperature_2m": "°C", "apparent_temperature": "°C", "wind_speed_10m": "km/h"},
    "current": {
        "temperature_2m": 31.2, "apparent_temperature": 35.0, "relative_humidity_2m": 48,
        "weather_code": 2, "wind_speed_10m": 9.4,
    },
    "daily": {
        "time": ["2026-11-10", "2026-11-11"],
        "weather_code": [2, 63],
        "temperature_2m_max": [33.0, 29.5],
        "temperature_2m_min": [20.1, 19.0],
        "precipitation_probability_max": [5, None],
    },
}


def test_get_weather_formats(monkeypatch):
    seen = {}

    def handler(req):
        if "geocoding" in req.url.host:
            seen["geo"] = dict(req.url.params)
            return httpx.Response(200, json=GEO)
        seen["fc"] = dict(req.url.params)
        return httpx.Response(200, json=FORECAST)

    _mock_web(monkeypatch, handler)
    r = web.get_weather("Delhi", days=2)
    assert r.ok
    assert "Delhi, Delhi, India" in r.content
    assert "31.2" in r.content and "feels like 35.0" in r.content and "partly cloudy" in r.content
    assert "humidity 48%" in r.content and "wind 9.4 km/h" in r.content
    assert "high 33.0" in r.content and "low 20.1" in r.content and "precipitation chance 5%" in r.content
    assert "rain" in r.content and "n/a" in r.content
    assert seen["fc"]["forecast_days"] == "2" and seen["geo"]["name"] == "Delhi"


def test_get_weather_clamps_days(monkeypatch):
    seen = {}

    def handler(req):
        if "geocoding" in req.url.host:
            return httpx.Response(200, json=GEO)
        seen["days"] = req.url.params["forecast_days"]
        return httpx.Response(200, json=FORECAST)

    _mock_web(monkeypatch, handler)
    web.get_weather("Delhi", days=30)
    assert seen["days"] == "7"


def test_get_weather_unknown_place(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(200, json={}))
    r = web.get_weather("Nowhereville")
    assert not r.ok and "Nowhereville" in r.content


def test_get_weather_comma_fallback(monkeypatch):
    def handler(req):
        if "geocoding" in req.url.host:
            return httpx.Response(200, json=GEO if req.url.params["name"] == "Delhi" else {})
        return httpx.Response(200, json=FORECAST)

    _mock_web(monkeypatch, handler)
    assert web.get_weather("Delhi, India").ok


def test_get_weather_service_down(monkeypatch):
    _mock_web(monkeypatch, lambda req: httpx.Response(503))
    assert not web.get_weather("Delhi").ok


def test_wmo_codes_cover_all_forecast_codes():
    for code in (0, 1, 2, 3, 45, 48, 51, 61, 65, 71, 80, 95, 99):
        assert code in web._WMO


# ---------------------------------------------------------------- time parsing
NOW = datetime(2026, 11, 10, 14, 30, 0)  # a Tuesday


@pytest.mark.parametrize(
    "text, expected",
    [
        ("in 20 minutes", NOW + timedelta(minutes=20)),
        ("in an hour", NOW + timedelta(hours=1)),
        ("in half an hour", NOW + timedelta(minutes=30)),
        ("in two days", NOW + timedelta(days=2)),
        ("in 1 week", NOW + timedelta(weeks=1)),
        ("in ten mins", NOW + timedelta(minutes=10)),
        ("tomorrow 9am", datetime(2026, 11, 11, 9, 0)),
        ("tomorrow", datetime(2026, 11, 11, 9, 0)),
        ("tomorrow at 5:30 pm", datetime(2026, 11, 11, 17, 30)),
        ("at 18:30", datetime(2026, 11, 10, 18, 30)),
        ("at 9am", datetime(2026, 11, 11, 9, 0)),  # passed today, bare time rolls to tomorrow
        ("tonight", datetime(2026, 11, 10, 20, 0)),
        ("noon tomorrow", datetime(2026, 11, 11, 12, 0)),
        ("friday 3pm", datetime(2026, 11, 13, 15, 0)),
        ("next tuesday 10am", datetime(2026, 11, 17, 10, 0)),
        ("today 5pm", datetime(2026, 11, 10, 17, 0)),
        ("2026-11-10T18:30", datetime(2026, 11, 10, 18, 30)),
        ("2026-11-12 08:00:00", datetime(2026, 11, 12, 8, 0)),
    ],
)
def test_parse_when_valid(text, expected):
    assert prod.parse_when(text, NOW) == expected


@pytest.mark.parametrize(
    "text",
    [
        "", "   ", "later", "soon", "sometime next year", "in 500 years", "in 0 minutes",
        "in 400 days", "at 25:00", "at 13pm", "today", "today 9am",
        "2026-11-09T10:00", "2026-11-10T14:30",
    ],
)
def test_parse_when_invalid(text):
    with pytest.raises(prod.WhenError):
        prod.parse_when(text, NOW)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("day after tomorrow", datetime(2026, 11, 12, 9, 0)),
        ("the day after tomorrow at 4pm", datetime(2026, 11, 12, 16, 0)),
        ("13 nov", datetime(2026, 11, 13, 9, 0)),
        ("13th of November 2026 at 6pm", datetime(2026, 11, 13, 18, 0)),
        ("Nov 13, 2026 8:15am", datetime(2026, 11, 13, 8, 15)),
        ("December 1st", datetime(2026, 12, 1, 9, 0)),
        ("5 jan", datetime(2027, 1, 5, 9, 0)),  # already passed this year
        ("13/11/2026 3pm", datetime(2026, 11, 13, 15, 0)),  # day first
        ("2026-11-13 8pm", datetime(2026, 11, 13, 20, 0)),
        ("saturday night", datetime(2026, 11, 14, 20, 0)),
        ("tomorrow afternoon", datetime(2026, 11, 11, 15, 0)),
        ("this weekend", datetime(2026, 11, 14, 9, 0)),
        ("next week", datetime(2026, 11, 17, 9, 0)),
    ],
)
def test_parse_when_understands_dates(text, expected):
    assert prod.parse_when(text, NOW) == expected


@pytest.mark.parametrize("text", ["yesterday 9am", "last week", "31 feb", "5 nov 2026", "32/01/2027"])
def test_parse_when_refuses_the_past_and_the_impossible(text):
    with pytest.raises(prod.WhenError):
        prod.parse_when(text, NOW)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("today", (2026, 11, 10)), ("tonight", (2026, 11, 10)), ("tomorrow", (2026, 11, 11)),
        ("next friday", (2026, 11, 13)), ("this Saturday night", (2026, 11, 14)),
        ("13 nov", (2026, 11, 13)), ("2026-12-24", (2026, 12, 24)), ("in 3 days", (2026, 11, 13)),
        ("the weekend", (2026, 11, 14)), ("day after tomorrow", (2026, 11, 12)),
    ],
)
def test_parse_day(text, expected):
    assert prod.parse_day(text, NOW) == date(*expected)


@pytest.mark.parametrize("text", ["", "whenever", "yesterday", "2026-11-09", "sometime"])
def test_parse_day_refuses(text):
    with pytest.raises(prod.WhenError):
        prod.parse_day(text, NOW)


def test_weekend_on_a_weekend_day():
    saturday, sunday = datetime(2026, 11, 14, 10, 0), datetime(2026, 11, 15, 10, 0)
    assert prod.parse_day("this weekend", saturday) == date(2026, 11, 14)
    assert prod.parse_day("this weekend", sunday) == date(2026, 11, 15)
    assert prod.parse_day("next weekend", saturday) == date(2026, 11, 21)


def test_parse_when_midnight_edge():
    late = datetime(2026, 11, 10, 23, 59)
    assert prod.parse_when("at 12:05 am", late) == datetime(2026, 11, 11, 0, 5)


# ---------------------------------------------------------------- reminders
def test_reminder_lifecycle(ctx):
    r = prod.set_reminder("call mum", "in 20 minutes", ctx=ctx)
    assert r.ok, r.content
    rid = r.data["id"]
    assert "call mum" in prod.list_reminders(ctx=ctx).content

    due_at = datetime.fromisoformat(r.data["due_at"])
    assert prod.due_reminders(ctx.memory, due_at - timedelta(minutes=1)) == []
    due = prod.due_reminders(ctx.memory, due_at + timedelta(seconds=1))
    assert [d["id"] for d in due] == [rid] and due[0]["message"] == "call mum"

    prod.mark_reminder_fired(ctx.memory, rid)
    assert prod.due_reminders(ctx.memory, due_at + timedelta(hours=1)) == []
    assert "No pending" in prod.list_reminders(ctx=ctx).content
    assert "[done]" in prod.list_reminders(ctx=ctx, include_done=True).content


def test_reminder_cancel(ctx):
    rid = prod.set_reminder("water plants", "tomorrow 9am", ctx=ctx).data["id"]
    assert prod.cancel_reminder(rid, ctx=ctx).ok
    assert prod.due_reminders(ctx.memory, datetime.now() + timedelta(days=3)) == []
    assert not prod.cancel_reminder(rid, ctx=ctx).ok  # already cancelled
    assert not prod.cancel_reminder(9999, ctx=ctx).ok
    assert "[cancelled]" in prod.list_reminders(ctx=ctx, include_done=True).content


def test_reminder_bad_time_gives_formats_and_now(ctx):
    r = prod.set_reminder("x", "whenever", ctx=ctx)
    assert not r.ok and "Hint" in r.content and "current local time" in r.content and "in 20 minutes" in r.content
    past = prod.set_reminder("x", "2020-01-01T10:00", ctx=ctx)
    assert not past.ok and "past" in past.content
    assert "No pending" in prod.list_reminders(ctx=ctx).content


def test_reminder_empty_message(ctx):
    assert not prod.set_reminder("  ", "in 5 minutes", ctx=ctx).ok


def test_productivity_without_memory():
    bare = ToolContext(settings=SimpleNamespace(), memory=None)
    for result in (
        prod.add_note("x", ctx=bare), prod.list_notes(ctx=bare), prod.search_notes("x", ctx=bare),
        prod.delete_note(1, ctx=bare), prod.set_reminder("x", "in 5 minutes", ctx=bare),
        prod.list_reminders(ctx=bare), prod.cancel_reminder(1, ctx=bare),
    ):
        assert not result.ok and "not available" in result.content


# ---------------------------------------------------------------- notes
def test_notes_lifecycle(ctx):
    a = prod.add_note("Buy oat milk and eggs", ctx=ctx, title="Groceries")
    b = prod.add_note("Idea: voice-controlled lights", ctx=ctx)
    assert a.ok and b.ok and a.data["id"] != b.data["id"]

    listing = prod.list_notes(ctx=ctx)
    assert "Groceries" in listing.content and "lights" in listing.content
    assert listing.content.index("lights") < listing.content.index("Groceries")  # newest first
    assert len(prod.list_notes(ctx=ctx, limit=1).data) == 1

    hit = prod.search_notes("MILK eggs", ctx=ctx)
    assert hit.ok and "Groceries" in hit.content and "lights" not in hit.content
    assert prod.search_notes("zebra", ctx=ctx).data == []

    assert prod.delete_note(a.data["id"], ctx=ctx).ok
    assert "Groceries" not in prod.list_notes(ctx=ctx).content
    assert not prod.delete_note(a.data["id"], ctx=ctx).ok
    assert not prod.add_note("  ", ctx=ctx).ok


def test_tool_specs_and_risk():
    for fn in (prod.delete_note, system.close_app, system.lock_screen):
        assert fn.__tool_spec__.risk is Risk.CONFIRM
    assert "ctx" not in prod.add_note.__tool_spec__.parameters["properties"]
    assert prod.set_reminder.__tool_spec__.group == "core"
    assert web.web_search.__tool_spec__.parameters["properties"]["recency"]["enum"] == ["any", "day", "week", "month"]
    assert system.media_key.__tool_spec__.parameters["properties"]["action"]["enum"][0] == "play_pause"
    assert system.close_app.__tool_spec__.confirmation_text({"name": "notepad"}).count("notepad")


# ---------------------------------------------------------------- apps
APPS = [
    system.AppEntry("Google Chrome", "lnk", r"C:\sm\Google Chrome.lnk"),
    system.AppEntry("Microsoft Word", "lnk", r"C:\sm\Word.lnk"),
    system.AppEntry("WordPad", "exe", "write.exe"),
    system.AppEntry("Visual Studio Code", "lnk", r"C:\sm\Code.lnk"),
    system.AppEntry("Spotify", "uwp", "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"),
    system.AppEntry("Notepad", "exe", "notepad.exe"),
    system.AppEntry("Calculator", "exe", "calc.exe"),
]


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("chrome", "Google Chrome"),
        ("Google Chrome browser", "Google Chrome"),
        ("chrom", "Google Chrome"),
        ("word", "Microsoft Word"),  # not WordPad
        ("code", "Visual Studio Code"),
        ("spotify", "Spotify"),
        ("notepad", "Notepad"),
        ("calc", "Calculator"),
    ],
)
def test_resolve_app(spoken, expected):
    best, _ = system.resolve_app(spoken, APPS)
    assert best is not None and best.name == expected


def test_resolve_app_no_match_gives_candidates():
    best, cands = system.resolve_app("photoshop", APPS)
    assert best is None
    best, cands = system.resolve_app("pad", APPS)
    assert best is None or best.name == "WordPad"


@pytest.fixture
def windows(monkeypatch, fake_platform):
    """A fake OS backend plus a fixed app catalogue (the name is historical: the tests below
    are platform-neutral and run identically on every OS)."""
    monkeypatch.setattr(system, "_catalog", lambda refresh=False: APPS)
    return fake_platform


def test_open_app_launches_best_match(monkeypatch, windows):
    launched = []
    monkeypatch.setattr(system, "_launch", launched.append)
    r = system.open_app("chrome")
    assert r.ok and launched[0].name == "Google Chrome"


def test_open_app_unknown_returns_candidates(monkeypatch, windows):
    monkeypatch.setattr(system, "_launch", lambda e: pytest.fail("must not launch"))
    r = system.open_app("Spotfy2000x")
    assert not r.ok and "Hint" in r.content


def test_open_app_launch_error(monkeypatch, windows):
    def boom(entry):
        raise OSError("denied")

    monkeypatch.setattr(system, "_launch", boom)
    assert not system.open_app("notepad").ok


def test_launch_delegates_to_the_platform(fake_platform):
    entry = system.AppEntry("A", "lnk", "x.lnk")
    system._launch(entry)
    assert fake_platform.launched == [entry]


def test_open_app_reports_unsupported_platform(monkeypatch, windows):
    def refuse(entry):
        raise PlatformUnsupported("opening apps is not supported here")

    monkeypatch.setattr(windows, "launch_app", refuse)
    r = system.open_app("chrome")
    assert not r.ok and "not supported here" in r.content


def test_list_installed_apps_query(monkeypatch, windows):
    r = system.list_installed_apps("chrome")
    assert "Google Chrome" in r.content and "Spotify" not in r.content
    assert "Spotify" in system.list_installed_apps().content


def test_catalog_drops_junk_entries_and_survives_scan_errors(fake_platform):
    fake_platform.apps = [
        system.AppEntry("Foo", "lnk", "foo.lnk"),
        system.AppEntry("Uninstall Foo", "lnk", "u.lnk"),
        system.AppEntry("Foo Readme", "lnk", "r.lnk"),
    ]
    assert [e.name for e in system._catalog(refresh=True)] == ["Foo"]

    def boom():
        raise OSError("scan failed")

    fake_platform.scan_apps = boom
    assert system._catalog(refresh=True) == []


# -- close_app
def _fake_windows(monkeypatch, windows, closes_on_post=True):
    """windows: list of (proc, title, pid, hwnd). Posting WM_CLOSE removes the window
    unless closes_on_post is False (an app that keeps a prompt open)."""
    state = list(windows)
    posted: list[int] = []

    def post(hwnd):
        posted.append(hwnd)
        if closes_on_post:
            state[:] = [w for w in state if w[3] != hwnd]

    monkeypatch.setattr(system, "_windows", lambda: list(state))
    monkeypatch.setattr(system, "_post_close", post)
    monkeypatch.setattr(system.time, "sleep", lambda s: None)
    return posted


def test_close_app_closes_only_that_apps_windows(monkeypatch, windows):
    posted = _fake_windows(monkeypatch, [
        ("notepad.exe", "Untitled - Notepad", 1, 101),
        ("notepad.exe", "todo.txt - Notepad", 1, 102),
        ("chrome.exe", "Notepad alternatives - Google Chrome", 2, 201),
    ])
    r = system.close_app("notepad")
    assert r.ok and r.data["closed"] is True
    assert posted == [101, 102]  # the Chrome tab that mentions the word is left alone


def test_close_app_in_shared_host_closes_one_window(monkeypatch, windows):
    posted = _fake_windows(monkeypatch, [
        ("ApplicationFrameHost.exe", "Calculator", 5, 301),
        ("ApplicationFrameHost.exe", "Settings", 5, 302),
        ("explorer.exe", "Program Manager", 6, 401),
        ("explorer.exe", "Downloads - File Explorer", 6, 402),
    ])
    r = system.close_app("calculator")
    assert r.ok and posted == [301]
    posted.clear()
    r = system.close_app("file explorer")
    assert r.ok and posted == [402]  # never the desktop shell


def test_close_app_by_process_name(monkeypatch, windows):
    posted = _fake_windows(monkeypatch, [
        ("chrome.exe", "YouTube - Google Chrome", 2, 201),
        ("chrome.exe", "Inbox - Google Chrome", 2, 202),
        ("Spotify.exe", "Spotify Premium", 3, 203),
    ])
    assert system.close_app("chrome").ok and posted == [201, 202]


def test_close_app_still_open(monkeypatch, windows):
    _fake_windows(monkeypatch, [("notepad.exe", "Untitled - Notepad", 1, 101)], closes_on_post=False)
    monkeypatch.setattr(system.time, "monotonic", iter(range(0, 1000, 5)).__next__)
    r = system.close_app("notepad")
    assert r.ok and r.data["closed"] is False and "Still open" in r.content


def test_close_app_not_running(monkeypatch, windows):
    _fake_windows(monkeypatch, [("chrome.exe", "x - Google Chrome", 1, 1)])
    r = system.close_app("notepad")
    assert not r.ok and "Google Chrome" in r.content


def test_close_app_never_kills_a_process(monkeypatch, windows):
    _fake_windows(monkeypatch, [("notepad.exe", "Untitled - Notepad", 1, 101)])

    def no_kill(*a, **k):
        raise AssertionError("close_app must not run a process-kill command")

    monkeypatch.setattr(system.subprocess, "run", no_kill)
    assert system.close_app("notepad").ok


# -- volume, media, lock, clipboard, url
def test_set_volume(windows):
    r = system.set_volume(40)
    assert r.ok and r.data == {"percent": 40, "muted": False} and windows.volume == (40, False)


@pytest.mark.parametrize("bad", [-1, 101, 1000])
def test_set_volume_range(windows, bad):
    assert not system.set_volume(bad).ok
    assert windows.volume == (50, False)  # untouched


def test_get_volume_and_mute(windows):
    windows.volume = (25, True)
    assert system.get_volume().data == {"percent": 25, "muted": True}
    assert "(muted)" in system.set_mute(True).content


def test_volume_failure_is_clean(windows):
    windows.volume_error = OSError("no audio endpoint")
    r = system.get_volume()
    assert not r.ok and "no audio endpoint" in r.content
    assert not system.set_volume(10).ok and not system.set_mute(True).ok


def test_volume_unsupported_message_is_passed_on(windows):
    windows.volume_error = PlatformUnsupported("volume control needs wpctl: install wireplumber")
    assert "install wireplumber" in system.get_volume().content


def test_media_key(windows):
    assert system.media_key("next").ok and system.media_key("play_pause").ok
    assert windows.keys == ["next", "play_pause"]


def test_media_key_failure(windows):
    windows.key_error = PlatformUnsupported("media keys need playerctl: install playerctl")
    r = system.media_key("stop")
    assert not r.ok and "install playerctl" in r.content


def test_lock_screen(windows):
    assert system.lock_screen().ok and windows.locked == 1


def test_lock_screen_failure(windows):
    windows.lock_error = OSError("no session")
    assert not system.lock_screen().ok


def test_list_running_apps_unsupported(windows):
    windows.windows_error = PlatformUnsupported("listing windows is not supported in a Wayland session")
    r = system.list_running_apps()
    assert not r.ok and "Wayland" in r.content
    r = system.close_app("notepad")
    assert not r.ok and "Wayland" in r.content


def test_list_running_apps(windows):
    windows.window_rows = [("a.exe", "A one", 1, 1), ("a.exe", "A two", 1, 2), ("b.exe", "B", 2, 3)]
    r = system.list_running_apps()
    assert r.ok and "a.exe: A one | A two" in r.content and "b.exe: B" in r.content


def test_clipboard(monkeypatch):
    import pyperclip

    store = {"v": "hello"}
    monkeypatch.setattr(pyperclip, "paste", lambda: store["v"])
    monkeypatch.setattr(pyperclip, "copy", lambda t: store.update(v=t))
    monkeypatch.setattr(system.plat.current(), "clipboard_problem", lambda: None)
    assert system.get_clipboard().content == "hello"
    assert system.set_clipboard("new").ok and store["v"] == "new"
    store["v"] = ""
    assert "empty" in system.get_clipboard().content


def test_clipboard_missing_helper_names_the_package(fake_platform):
    fake_platform.clipboard_text = "the clipboard needs xclip or xsel (X11): install xclip"
    for r in (system.get_clipboard(), system.set_clipboard("x")):
        assert not r.ok and "install xclip" in r.content


def test_open_url(monkeypatch):
    opened = []
    monkeypatch.setattr(system, "_open_in_browser", lambda u: opened.append(u) or True)
    assert system.open_url("https://example.com/a").ok and opened == ["https://example.com/a"]
    for bad in ("file:///C:/x", "javascript:alert(1)", "example.com", "ftp://x.com"):
        assert not system.open_url(bad).ok
    assert opened == ["https://example.com/a"]


def test_open_url_goes_through_the_platform(fake_platform):
    assert system.open_url("https://example.com/b").ok
    assert fake_platform.urls == ["https://example.com/b"]
    fake_platform.open_error = PlatformUnsupported("opening files and links needs xdg-open: install xdg-utils")
    r = system.open_url("https://example.com/c")
    assert not r.ok and "install xdg-utils" in r.content


# ---------------------------------------------------------------- real, read-only
def test_get_time_real():
    r = system.get_time()
    assert r.ok and str(datetime.now().year) in r.content and "UTC" in r.content


def test_system_status_real(monkeypatch):
    monkeypatch.setattr(system, "_network_connected", lambda: True)  # keep the test offline
    r = system.system_status()
    assert r.ok
    for label in ("CPU:", "RAM:", "Battery:", "Disks:", "Top memory:", "Uptime:", "Network:"):
        assert label in r.content


def test_web_search_drops_sponsored_redirects_and_keeps_the_count(ddgs):
    ddgs.results = [
        {"title": "Ad", "href": "https://www.bing.com/aclick?ld=abc&u=xyz", "body": "sponsored"},
        {"title": "Ad two", "href": "https://duckduckgo.com/y.js?ad_domain=x", "body": "sponsored"},
        *({"title": f"Real {i}", "href": f"https://example.com/{i}", "body": "text"} for i in range(4)),
    ]
    r = web.web_search("flights delhi mumbai", max_results=3)
    assert r.ok and "aclick" not in r.content and "y.js" not in r.content
    assert [x["title"] for x in r.data] == ["Real 0", "Real 1", "Real 2"]


def test_web_search_retries_a_spurious_no_results_once(ddgs, monkeypatch):
    monkeypatch.setattr(web.time, "sleep", lambda s: None)
    answers = [web.DDGSException("No results found."), [{"title": "T", "href": "https://example.com", "body": "b"}]]

    def flaky(query, n, timelimit):
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(web, "_ddgs_text", flaky)
    assert web.web_search("something").ok and answers == []
    answers[:] = [web.DDGSException("No results found."), web.DDGSException("No results found.")]
    r = web.web_search("nothing at all")
    assert not r.ok and "no results for" in r.content and "internet" not in r.content
