"""Desktop-control policy and key-chord parsing: pure tables, no OS, no UI."""

from __future__ import annotations

import pytest

from jarvis.desktop import policy
from jarvis.platform.base import UiElement, WindowInfo
from jarvis.platform.keys import ChordError, parse_chord


def el(name="", role="button", automation_id="", password=False, enabled=True):
    return UiElement(ref=1, role=role, name=name, automation_id=automation_id, is_password=password, enabled=enabled)


def win(process="notepad.exe", title="Untitled - Notepad", klass="Notepad", pid=4242, elevated=False):
    return WindowInfo(title=title, process=process, class_name=klass, pid=pid, handle=100, elevated=elevated)


# ---------------------------------------------------------------- clicks (what the control is called)
CLICKS = [
    # (role, name, automation id, expected)
    ("button", "Seven", "num7Button", "safe"),
    ("button", "Plus", "plusButton", "safe"),
    ("button", "Equals", "equalButton", "safe"),
    ("button", "Cancel", "", "safe"),
    ("button", "Search", "", "safe"),
    ("button", "Add to cart", "", "safe"),
    ("link", "Read more", "", "safe"),
    ("tab item", "Settings", "", "safe"),
    ("menu item", "Copy", "", "safe"),
    ("list item", "Downloads", "", "safe"),
    ("checkbox", "Show hidden files", "", "safe"),
    ("button", "Buy now", "", "ask"),
    ("button", "Place your order", "", "ask"),
    ("button", "Pay $19.99", "", "ask"),
    ("button", "Send", "", "ask"),
    ("button", "Post", "", "ask"),
    ("button", "Delete", "", "ask"),
    ("button", "Delete account", "", "ask"),
    ("button", "Sign in", "", "ask"),
    ("button", "Log in with Google", "", "ask"),
    ("button", "I agree", "", "ask"),
    ("button", "Accept all", "", "ask"),
    ("button", "Change password", "", "ask"),
    ("button", "Transfer funds", "", "ask"),
    ("button", "Confirm", "", "ask"),
    ("button", "OK", "", "ask"),  # a bare OK hides what it confirms
    ("button", "Yes", "", "ask"),
    ("button", "Button", "SendMessageButton", "ask"),  # the label is vague, the automation id is not
    ("checkbox", "I accept the terms and conditions", "", "ask"),
    ("link", "Unsubscribe", "", "ask"),
    ("menu item", "Empty Recycle Bin", "", "ask"),  # desktop-only additions to the browser classifier
    ("button", "Shut down", "", "ask"),
    ("button", "Restart now", "", "ask"),
    ("button", "Uninstall", "", "ask"),
    ("button", "Format disk", "", "ask"),
    ("menu item", "Sign out", "", "ask"),
    ("button", "Overwrite", "", "ask"),
    ("menu item", "Run as administrator", "", "ask"),
    ("button", "Reset zoom", "", "safe"),
    ("button", "Subscribe", "", "ask"),
    ("button", "Share", "", "ask"),
]


@pytest.mark.parametrize("role,name,aid,expected", CLICKS)
def test_click_table(role, name, aid, expected):
    assert policy.classify_element_click(el(name, role, aid)).kind == expected


# ---------------------------------------------------------------- typing into fields
FIELDS = [
    ("edit", "Search", "", False, "hello", "safe"),
    ("edit", "File name:", "1001", False, "report.txt", "safe"),
    ("edit", "Message", "", False, "see you at 5", "safe"),
    ("edit", "Password", "", False, "hunter2", "refuse"),
    ("edit", "Enter password", "", False, "x", "refuse"),
    ("edit", "Confirm passcode", "", False, "x", "refuse"),
    ("edit", "Card number", "", False, "1234", "refuse"),
    ("edit", "CVV", "", False, "123", "refuse"),
    ("edit", "Security code", "", False, "123", "refuse"),
    ("edit", "One-time code", "", False, "123456", "refuse"),
    ("edit", "Name", "passwordBox", False, "x", "refuse"),  # by automation id
    ("edit", "Name", "", True, "x", "refuse"),  # UIA IsPassword
    ("edit", "Notes", "", False, "4111 1111 1111 1111", "refuse"),  # a Luhn-valid card number anywhere
    ("edit", "Notes", "", False, "call 555 0100 tomorrow", "safe"),
    ("document", "Text editor", "", False, "print('hi')", "safe"),
]


@pytest.mark.parametrize("role,name,aid,password,text,expected", FIELDS)
def test_typing_table(role, name, aid, password, text, expected):
    verdict = policy.classify_element_typing(el(name, role, aid, password), text)
    assert verdict.kind == expected


def test_typing_without_a_known_field_still_refuses_card_numbers():
    assert policy.classify_element_typing(None, "4111 1111 1111 1111").kind == "refuse"
    assert policy.classify_element_typing(None, "hello").kind == "safe"


# ---------------------------------------------------------------- windows
def test_window_kinds():
    assert policy.window_kind(win("cmd.exe", "C:\\Windows\\system32\\cmd.exe", "ConsoleWindowClass")) == "terminal"
    assert policy.window_kind(win("WindowsTerminal.exe", "Windows PowerShell", "CASCADIA_HOSTING_WINDOW_CLASS")) == "terminal"
    assert policy.window_kind(win("pwsh.exe", "pwsh", "whatever")) == "terminal"
    assert policy.window_kind(win("gnome-terminal-", "user@host: ~", "Gnome-terminal")) == "terminal"
    assert policy.window_kind(win("explorer.exe", "Run", "#32770")) == "run dialog"
    assert policy.window_kind(win("explorer.exe", "Run the numbers - Excel", "XLMAIN")) == "normal"
    assert policy.window_kind(win("consent.exe", "User Account Control", "Credential Dialog Xaml Host")) == "security"
    assert policy.window_kind(win("Calculator.exe", "User Account Control", "x")) == "security"
    assert policy.window_kind(win("chrome.exe", "Inbox - Gmail", "Chrome_WidgetWin_1")) == "normal"


def block(window, **kw):
    kw.setdefault("own_pids", {1})
    return policy.window_block(window, **kw)


def test_normal_window_is_not_blocked():
    assert block(win()) is None


def test_own_window_is_refused():
    verdict = block(win(pid=1))
    assert verdict.kind == "refuse" and "Jarvis's own window" in verdict.reason


def test_elevated_window_is_refused_from_a_normal_process():
    verdict = block(win("regedit.exe", "Registry Editor", elevated=True))
    assert verdict.kind == "refuse" and "administrator" in verdict.reason and "normal rights" in verdict.reason


def test_elevated_window_is_allowed_when_jarvis_is_elevated():
    assert block(win(elevated=True), self_elevated=True) is None


def test_security_surfaces_are_refused():
    assert block(win("consent.exe", "User Account Control")).kind == "refuse"
    assert block(win("chrome.exe", "Windows Security", "x")).kind == "refuse"


def test_secure_desktop_and_no_window_are_reported():
    gone = block(win(), input_desktop_ok=False)
    assert gone.kind == "refuse" and "UAC" in gone.reason
    assert block(None).kind == "refuse"


# ---------------------------------------------------------------- typing / Enter in command surfaces
def entry(window, text="hello", **kw):
    return policy.classify_text_entry(window, text, **kw)


def test_typing_into_ordinary_windows_is_safe():
    assert entry(win()).kind == "safe"
    assert entry(win(), "line one\nline two").kind == "safe"


@pytest.mark.parametrize(
    "window",
    [
        win("cmd.exe", "Command Prompt", "ConsoleWindowClass"),
        win("powershell.exe", "Windows PowerShell", "ConsoleWindowClass"),
        win("WindowsTerminal.exe", "Terminal", "CASCADIA_HOSTING_WINDOW_CLASS"),
        win("explorer.exe", "Run", "#32770"),
    ],
)
def test_typing_into_a_command_surface_asks_and_shows_the_text(window):
    verdict = entry(window, "del /q *.*")
    assert verdict.kind == "ask" and "del /q *.*" in verdict.reason


def test_typing_with_enter_in_a_terminal_says_so():
    verdict = entry(win("cmd.exe", "cmd", "ConsoleWindowClass"), "dir", submit=True)
    assert verdict.kind == "ask" and "press Enter" in verdict.reason


def test_an_elevated_window_asks_only_when_jarvis_is_elevated():
    window = win("notepad.exe", elevated=True)
    assert entry(window, self_elevated=True).kind == "ask"
    assert entry(window, self_elevated=False).kind == "safe"  # (window_block refuses it before this)


def test_submit_in_a_message_field_asks():
    chat = el("Type a message", "edit")
    assert entry(win("whatsapp.exe", "WhatsApp"), "hi", element=chat, submit=True).kind == "ask"
    search = el("Search", "edit")
    assert entry(win(), "hi", element=search, submit=True).kind == "safe"
    assert entry(win(), "hi", element=chat, submit=False).kind == "safe"


# ---------------------------------------------------------------- key chords
def chord(text, window=None, **kw):
    return policy.classify_chord(parse_chord(text), window or win(), **kw)


CHORDS = [
    ("ctrl+s", "safe"), ("ctrl+c", "safe"), ("ctrl+v", "safe"), ("ctrl+z", "safe"), ("ctrl+a", "safe"),
    ("alt+tab", "safe"), ("win+d", "safe"), ("escape", "safe"), ("tab", "safe"), ("f5", "safe"),
    ("ctrl+shift+t", "safe"), ("ctrl+t", "safe"), ("backspace", "safe"), ("ctrl+f", "safe"),
    ("down", "safe"), ("pagedown", "safe"), ("enter", "safe"), ("space", "safe"), ("ctrl+delete", "safe"),
    ("alt+f4", "ask"), ("ctrl+w", "ask"), ("ctrl+shift+w", "ask"), ("ctrl+f4", "ask"), ("ctrl+q", "ask"),
    ("delete", "ask"), ("shift+delete", "ask"), ("ctrl+shift+delete", "ask"), ("win+r", "ask"),
    ("win+l", "ask"),
    ("ctrl+alt+delete", "refuse"),
]


@pytest.mark.parametrize("text,expected", CHORDS)
def test_chord_table(text, expected):
    assert chord(text).kind == expected


def test_win_l_is_the_lock_screen_and_says_so():
    verdict = chord("win+l")
    assert verdict.kind == "ask" and "lock" in verdict.reason
    assert policy.is_win_l(parse_chord("Win+L")) and not policy.is_win_l(parse_chord("win+d"))


def test_enter_in_a_terminal_or_run_dialog_asks():
    cmd = win("cmd.exe", "cmd", "ConsoleWindowClass")
    assert chord("enter", cmd).kind == "ask"
    assert chord("enter", win("explorer.exe", "Run", "#32770")).kind == "ask"
    assert chord("ctrl+c", cmd).kind == "safe"
    assert chord("ctrl+enter", cmd).kind == "ask"


def test_enter_or_space_on_a_dangerous_focused_control_asks():
    delete = el("Delete", "button")
    verdict = chord("enter", focused=delete)
    assert verdict.kind == "ask" and "Delete" in verdict.reason
    assert chord("space", focused=delete).kind == "ask"
    assert chord("enter", focused=el("Seven", "button")).kind == "safe"
    assert chord("ctrl+enter", focused=delete).kind == "safe"  # not a plain activation


# ---------------------------------------------------------------- the chord parser
@pytest.mark.parametrize(
    "text,expected",
    [
        ("ctrl+s", ["ctrl", "s"]),
        ("Ctrl+Shift+T", ["ctrl", "shift", "t"]),
        ("shift+ctrl+t", ["ctrl", "shift", "t"]),  # modifiers are put in canonical order
        ("alt+tab", ["alt", "tab"]),
        ("ENTER", ["enter"]),
        ("return", ["enter"]),
        ("esc", ["escape"]),
        ("win+d", ["win", "d"]),
        ("windows+e", ["win", "e"]),
        ("control+alt+del", ["ctrl", "alt", "delete"]),
        ("pgdn", ["pagedown"]),
        ("f12", ["f12"]),
        ("ctrl+f24", ["ctrl", "f24"]),
        ("win", ["win"]),  # a lone modifier opens the Start menu
        ("ctrl++", ["ctrl", "plus"]),
        ("ctrl+-", ["ctrl", "minus"]),
        ("ctrl+,", ["ctrl", "comma"]),
        ("  alt + left  ", ["alt", "left"]),
        ("7", ["7"]),
        ("ctrl+shift", ["ctrl", "shift"]),
    ],
)
def test_valid_chords(text, expected):
    assert parse_chord(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "ctrl+", "+s", "ctrl+ctrl+s", "ctrl+banana", "ctrl+a+b", "hyper+x", "f25", "enterr", "ctrl+s+"],
)
def test_invalid_chords_are_rejected(text):
    with pytest.raises(ChordError):
        parse_chord(text)


def test_unknown_key_error_lists_the_valid_names():
    with pytest.raises(ChordError) as err:
        parse_chord("ctrl+banana")
    message = str(err.value)
    assert "banana" in message and "enter" in message and "f1-f24" in message and "modifiers" in message


def test_two_main_keys_get_a_helpful_message():
    with pytest.raises(ChordError, match="more than one non-modifier key"):
        parse_chord("a+b")
