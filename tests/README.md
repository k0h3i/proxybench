# Tests

The tests cover source preservation, review bindings, record validation, and scoring.
A binding connects an answer to its exact input artifacts.
Synthetic cases test failure behavior without contacting SEC or using model APIs.
No test result establishes historical extraction accuracy.

Use `fixtures/` for small synthetic or deliberately selected shareable examples. A fixture is fixed input used by a test. Keep downloaded corpora and private pilot labels under the ignored `data/` directory.

The implemented cases cover these behaviors:

- Preserve source locations through document preparation.
- Match logical subjects through reviewed source anchors.
- Distinguish missing values from explicit absence of a recommendation.
- Reject extra and duplicate predictions in the fragment metric.
- Preserve split votes and collective disclosures without invented detail.

Run the Python tests from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`review_runtime_cases.js` tests the actual JavaScript save, resume, and correction functions from the review template.
Its host supplies those function definitions to `runReviewCases(helpers, validation)`.
The implementation report records execution through the available JavaScript runtime.
These function tests do not replace browser interaction tests.

Run the focused regressions for reference admission, scoring reports, review drafts, and audit consistency:

```bash
PYTHONPATH=src:tests python3 -m unittest test_bindings test_benchmark test_review test_audit -v
```

`review_browser_cases.js` tests a newly rendered synthetic review page in a same-origin browser frame.
Call `runReviewBrowserCases(frame.contentWindow)` after the frame loads.
It exercises file import, rejection without state replacement, resume, export counts, and browser-storage rejection.
The test intercepts downloads and supplies browser `File` objects.
It does not exercise native file dialogs or native downloads.

The runner tests use fixed synthetic responses and bounded local subprocesses.
They cover date rules, capture integrity, timeouts, and output limits.
The local-input tests cover XML boundaries, sequence masks, and resource limits.

The historical pilot tests cover exact CPU training resume, source-value scoring, review bindings, and the fixed resource schedule.
The process tests cover live output, slow terminals, interrupts, and cleanup without GPU work.
Use the existing isolated environment to include the tensor tests, as shown in the [pilot guide](../docs/qwen35-4b-historical-pilot.md).

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_runner.py" -v
```
