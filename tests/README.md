# Tests

Add behavior tests as pipeline components are implemented. This scaffold contains no extraction tests yet. Test requirements come from the reviewed specification and calibration examples.

Use `fixtures/` for small synthetic or deliberately selected shareable examples. A fixture is fixed input used by a test. Keep downloaded corpora and private pilot labels under the ignored `data/` directory.

Prioritize tests for these behaviors:

- Preserve source locations through document preparation.
- Separate ballot subjects and reset context at fund boundaries.
- Distinguish missing values from explicit absence of a recommendation.
- Reject extra and duplicate predictions in the fragment metric.
- Preserve split votes and collective disclosures without invented detail.
