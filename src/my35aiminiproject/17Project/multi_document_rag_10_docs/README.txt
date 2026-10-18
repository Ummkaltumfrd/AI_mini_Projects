MULTI-DOCUMENT RAG — 10-FILE DEMO DATASET
================================================

All company names and numbers are fictional, created for testing.

FILES (10 total)
- 3 PDF
- 3 DOCX
- 2 TXT
- 2 PPTX

SUGGESTED CROSS-DOCUMENT TEST QUESTION
"What percentage of Atlas Energy's 2024 electricity consumption could be supplied by its solar generation, and how does that compare with the 2026 renewable-share target?"

Required evidence:
- 01_solar_output_2024.pdf: annual solar generation = 120 MWh
- 02_electricity_consumption.docx: annual electricity consumption = 300 MWh
- 03_energy_targets.txt: 2026 renewable-share target = at least 45%

Expected calculation:
120 / 300 * 100 = 40%.
This is 5 percentage points below the 45% target.

This is a genuine multi-document test: no single file contains all three values.

METADATA SUGGESTION
For every stored chunk, attach:
- filename: exact source filename
- section: heading associated with the chunk
- file_type: pdf / docx / txt / pptx

FILTER TESTS
1. Search only the solar report: where={"filename": "01_solar_output_2024.pdf"}
2. Search only the "Annual Consumption" section: where={"section": "Annual Consumption"}
3. Search all documents without a metadata filter.

DOC LAYOUT
Each document contains named sections/headings. For PPTX files, each section is a separate slide.
