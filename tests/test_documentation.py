"""Keep README and reproduction navigation valid in the delivered checkout."""

import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    def test_relative_links_resolve(self):
        documents = [REPO / "README.md", REPO / "evidence/README.md", *sorted((REPO / "docs").glob("*.md"))]
        for document in documents:
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                if "://" in target or target.startswith("mailto:"):
                    continue
                path, _, fragment = unquote(target).partition("#")
                resolved = (document.parent / path).resolve() if path else document
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(resolved.exists(), f"missing link: {resolved}")
                    if fragment and resolved.suffix == ".md":
                        headings = re.findall(r"^#{1,6}\s+(.+)$", resolved.read_text(), re.M)
                        anchors = {re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
                                   for heading in headings}
                        self.assertIn(fragment, anchors)

    def test_overview_is_a_self_contained_vector_bound_to_the_fixture(self):
        manifest = json.loads((REPO / "assets/overview.json").read_text())
        fixture = REPO / manifest["trajectory"]
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), manifest["trajectory_sha256"])
        root = ET.parse(REPO / "assets/overview.svg").getroot()
        self.assertEqual(root.attrib["viewBox"], "0 0 1160 340")
        self.assertNotIn("script", {element.tag.rsplit("}", 1)[-1] for element in root.iter()})


if __name__ == "__main__":
    unittest.main()
