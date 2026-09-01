"""Streamlit page smoke tests and prohibited-training-call audit."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
PAGES = [
    ROOT / "pages" / "1_Overview.py",
    ROOT / "pages" / "2_Incident_Profile_Explorer.py",
    ROOT / "pages" / "3_County_Pattern_Explorer.py",
    ROOT / "pages" / "4_Model_Evaluation.py",
    ROOT / "pages" / "5_Interactive_Profile_Explorer.py",
]


def assert_clean(app_test: AppTest, label: str) -> None:
    if len(app_test.exception):
        raise AssertionError(f"{label} raised: {[item.message for item in app_test.exception]}")


for page in PAGES:
    test = AppTest.from_file(str(page), default_timeout=60).run()
    assert_clean(test, page.name)
    print("PASS", page.name)

# Exercise the county model toggle and a state display filter.
county_test = AppTest.from_file(str(PAGES[2]), default_timeout=60).run()
county_test.radio[0].set_value("Conventional K-Means").run()
assert_clean(county_test, "county model toggle")
county_test.multiselect[0].set_value(["California"]).run()
assert_clean(county_test, "county state filter")

# Exercise the hypothetical matcher. This loads existing centroids and performs
# distance calculations only.
matcher_test = AppTest.from_file(str(PAGES[4]), default_timeout=60).run()
matcher_test.radio[0].set_value("Hypothetical situational scenario").run()
assert_clean(matcher_test, "hypothetical matcher")
matcher_test.radio[1].set_value("Select an existing county").run()
assert_clean(matcher_test, "existing-county context matcher")
matcher_test.radio[1].set_value("Customize context").run()
assert_clean(matcher_test, "customized context matcher")

prohibited_tokens = [
    "KMeans.fit(", "AgglomerativeClustering.fit(", "HDBSCAN.fit(",
    ".fit_predict(", "Skater(",
]
application_files = [ROOT / "app.py", ROOT / "data_loader.py", ROOT / "utils.py", *PAGES]
for path in application_files:
    source = path.read_text(encoding="utf-8")
    for token in prohibited_tokens:
        if token in source:
            raise AssertionError(f"Prohibited training token {token!r} in {path.name}")

print("PASS model-refit source audit")
