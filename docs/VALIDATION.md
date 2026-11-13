# Release validation

Packaging validation was performed on 13 November 2026.

- All 12 evaluation JSON files retained their source SHA-256 hashes, with 200 non-empty records per file and matching ordered headword/reference pairs across zero-shot and five-shot variants.
- All 25 packaged Python files passed syntax parsing.
- Four unit tests passed: test-set integrity, historical radical/substring scoring, prediction alignment and perfect ordinal agreement on a small artificial unit fixture.
- Nine archived command-line help entry points ran without generating model responses.
- Scoring smoke tests covered both historical and normalised-exact classification, and both historical and supplementary ROUGE profiles. Perfect-reference fixtures were mechanical checks, not model predictions or new experimental results. Fixtures are not included in this release.
- Two fine-tuning comparison figures and three supplementary figures were generated from included numerical summaries into a separate private validation directory.
- No API calls, model generation, Sentence-BERT weight downloads or scientific experiment reruns were performed.

The private historical ROUGE smoke test used the archived local zhconv installation. The public requirements declare the upstream zhconv package instead; vendored dependencies are not distributed. A universal fresh-environment installation test, GPU inference test and independent legal clearance are outside this validation scope.

Run `python tools/validate_release.py` to verify distributed checksums. Generating figures creates new output files; those generated outputs are not part of the original checksummed source inventory. This document reports packaging checks, not an independent verification of every scientific claim or a guarantee that remote services will reproduce historical outputs.
