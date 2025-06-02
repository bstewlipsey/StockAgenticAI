import os
import re
import pytest

# List of placeholder/dummy phrases to scan for
PLACEHOLDER_PATTERNS = [
    r"Not implemented",
    r"Dummy value",
    r"TODO:",
    r"FIXME:",
    r"Actual LLM response",
    r"Placeholder",
    r"stub",
    r"mocked output",
]

# Respect test mode flag from config_system.py
try:
    from config_system import TEST_MODE_ENABLED
except ImportError:
    TEST_MODE_ENABLED = False

def scan_file_for_placeholders(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return pattern
    return None

def get_files_to_scan():
    # Scan logs and report outputs, but not test/mocks
    files = []
    # Scan trading.log and all reports/*.txt
    if os.path.exists("trading.log"):
        files.append("trading.log")
    reports_dir = os.path.join(os.getcwd(), "reports")
    if os.path.isdir(reports_dir):
        for fname in os.listdir(reports_dir):
            if fname.endswith(".txt"):
                files.append(os.path.join(reports_dir, fname))
    return files

@pytest.mark.skipif(TEST_MODE_ENABLED, reason="Skip placeholder check in test mode.")
def test_no_placeholders_in_outputs():
    print("[test_placeholder_detection] Running placeholder scan...")
    files = get_files_to_scan()
    found = []
    for f in files:
        pattern = scan_file_for_placeholders(f)
        if pattern:
            found.append((f, pattern))
    assert not found, f"Placeholder values found in outputs: {found}"
