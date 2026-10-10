from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from newspulse.cleaning import ValidationReport
from newspulse.dataset_service import Dataset
from newspulse.ui.nlp_analysis import next_analysis_run_id


def test_dataset_state_displays_import_error_and_empty_state(tmp_path: Path) -> None:
    error_script = tmp_path / "invalid_dataset_page.py"
    error_script.write_text(
        "import pandas as pd\n"
        "from newspulse.cleaning import ValidationReport\n"
        "from newspulse.dataset_service import Dataset\n"
        "from newspulse.ui.components import render_dataset_state\n"
        "render_dataset_state(Dataset(pd.DataFrame(), ValidationReport(0, 0, 0, 0, ()), 'Could not parse CSV'))\n"
    )
    invalid = AppTest.from_file(str(error_script)).run()
    assert not invalid.exception
    assert any("Could not parse CSV" in item.value for item in invalid.error)

    empty_script = tmp_path / "empty_dataset_page.py"
    empty_script.write_text(
        "import pandas as pd\n"
        "from newspulse.cleaning import ValidationReport\n"
        "from newspulse.dataset_service import Dataset\n"
        "from newspulse.ui.components import render_dataset_state\n"
        "render_dataset_state(Dataset(pd.DataFrame(), ValidationReport(0, 0, 0, 0, ())))\n"
    )
    empty = AppTest.from_file(str(empty_script)).run()
    assert not empty.exception
    assert any("upload a valid" in item.value.casefold() for item in empty.info)


def test_analysis_rerun_uses_a_new_cache_nonce() -> None:
    assert next_analysis_run_id(0, False) == 0
    assert next_analysis_run_id(0, True) == 1
    assert next_analysis_run_id(4, True) == 5
