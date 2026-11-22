import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.environ.get("MVRDC_DATA", os.path.join(ROOT, "data", "MVRDC"))


@pytest.fixture(scope="session")
def data_root():
    if not os.path.isdir(os.path.join(DATA, "Instances")):
        pytest.skip("instances not found; run scripts/fetch_data.sh or set MVRDC_DATA")
    return DATA
