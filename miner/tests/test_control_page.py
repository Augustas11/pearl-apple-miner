import json
import re
from html.parser import HTMLParser

from pmk_miner.control_page import render_control_page


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.scripts = []
        self.visible = []
        self._hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "link" and data.get("href"):
            self.links.append(data["href"])
        if tag == "script" and data.get("src"):
            self.scripts.append(data["src"])
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data):
        if not self._hidden_depth:
            self.visible.append(data)


def page(wallet="prl1wallet", secret="a" * 64):
    return render_control_page(wallet, secret, 47811, "Aug's Mac <Air>")


def test_control_page_has_required_local_ui_and_relative_api_calls():
    html = page()
    assert "Malibu Pearl <span>&middot; This Mac</span>" in html
    assert "Starting up" in html
    assert "Paused on battery" in html
    assert "Paused, Mac is hot" in html
    assert "M5 fast path" in html
    assert "Open on HeroMiners" in html
    assert "Miner on GitHub" in html
    assert 'fetch("./api/status"' in html
    assert 'fetch("./api/control"' in html
    assert '"X-Pearl-Local": PMK.localSecret' in html
    assert "stats_address?address=" in html
    assert '$("heroLink").href = "https://pearl.herominers.com/"' in html
    assert "setInterval(pollStatus, 2000)" in html
    assert "setInterval(fetchEarnings, 60000)" in html


def test_control_page_json_escapes_wallet_secret_and_computer_name():
    wallet = 'prl1"</script><img src=x onerror=alert(1)>'
    secret = "0123456789abcdef" * 4
    html = page(wallet, secret)
    config = re.search(r"const PMK = (\{.*?\});", html, re.S)
    assert config is not None
    data = json.loads(config.group(1))
    assert data["wallet"] == wallet
    assert data["localSecret"] == secret
    assert data["computerName"] == "Aug's Mac <Air>"
    assert "</script><img" not in html


def test_control_page_does_not_render_secret_as_visible_text():
    secret = "f" * 64
    parser = ResourceParser()
    parser.feed(page(secret=secret))
    assert secret not in " ".join(parser.visible)


def test_control_page_has_no_external_script_and_only_google_font_resource_links():
    parser = ResourceParser()
    parser.feed(page())
    assert parser.scripts == []
    assert parser.links == []
    resource_urls = re.findall(r'url\("?(https?://[^")]+)', page())
    assert resource_urls
    assert all(url.startswith("https://fonts.googleapis.com/") for url in resource_urls)
