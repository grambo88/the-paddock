"""
check_urls.py
=============
Tests every URL in the 2025 Saratoga meet window and reports
which dates return 200 (page exists) vs other status codes.

Usage:
  python check_urls.py
"""

import time
import requests
from datetime import date, timedelta

BASE_URL  = "https://entries.horseracingnation.com/entries-results/saratoga"
START     = date(2025, 7, 9)
END       = date(2025, 9, 1)
HEADERS   = {"User-Agent": "Mozilla/5.0"}

found     = []
not_found = []
errors    = []

total = (END - START).days + 1
print(f"Checking {total} dates from {START} to {END}...\n")

d = START
while d <= END:
    ds  = d.isoformat()
    url = f"{BASE_URL}/{ds}"
    try:
        r = requests.head(url, timeout=10, allow_redirects=True, headers=HEADERS)
        status = r.status_code
        if status == 200:
            found.append(ds)
            print(f"  ✓  {ds}  →  {status}")
        else:
            not_found.append(ds)
            print(f"  ✗  {ds}  →  {status}")
    except requests.RequestException as e:
        errors.append(ds)
        print(f"  !  {ds}  →  ERROR: {e}")

    time.sleep(0.8)   # polite pause
    d += timedelta(days=1)

print(f"\n{'─'*40}")
print(f"Found (200):   {len(found)}")
print(f"Not found:     {len(not_found)}")
print(f"Errors:        {len(errors)}")
print(f"\nRace days with pages:")
for ds in found:
    print(f"  {ds}")