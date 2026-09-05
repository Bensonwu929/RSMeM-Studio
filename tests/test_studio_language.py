"""Protect query-first onboarding and the boundary around translated UI copy."""

from html.parser import HTMLParser

import app
from studio_i18n import TRANSLATIONS
from studio_ui import hero_html, walkthrough_html


def test_query_precedes_optional_cases_in_the_rendered_layout():
    config = app.demo.get_config_file()

    def flatten(node):
        yield node["id"]
        for child in node.get("children", []):
            yield from flatten(child)

    order = list(flatten(config["layout"]))
    assert order.index(app.query_input._id) < order.index(app.example_input._id)
    picker = next(c for c in config["components"] if c["props"].get("elem_id") == "example-picker")
    assert picker["props"]["open"] is False
    assert "Need an idea?" in picker["props"]["label"]


def test_live_answers_and_evidence_are_not_translation_scopes():
    for component in (app.final_output, app.hkg_output, app.evaluation_output, app.decision_output):
        assert "rsmem-localized" not in component.elem_classes
    assert "rsmem-localized" in app.query_input.elem_classes


class CopyCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_code = False
        self.copy = []

    def handle_starttag(self, tag, attrs):
        if tag == "code":
            self.in_code = True

    def handle_endtag(self, tag):
        if tag == "code":
            self.in_code = False

    def handle_data(self, data):
        text = data.strip()
        if not self.in_code and any(char.isalpha() for char in text):
            self.copy.append(text)


def test_hero_and_walkthrough_have_translations_for_all_explanatory_copy():
    collector = CopyCollector()
    collector.feed(hero_html() + walkthrough_html())
    untranslated = set(collector.copy) - TRANSLATIONS.keys() - {"NDVI", "LST", "TVDI", "2021.07.12 · NDVI"}
    assert not untranslated
