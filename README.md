# Project WISE — Water Infrastructure Siting for Employment

Project WISE builds a data-driven framework to help **prioritize and site rural water infrastructure** (e.g., wells, check dams) so that interventions are both **socially targeted** (high need for water + employment) and **hydrologically effective**.

## Core idea (two-model framework)
- **Model 1: Priority Map**  
  Identifies high-need areas using socio-economic + employment + water-stress signals, augmented with **satellite imagery + ML** to overcome outdated/low-resolution administrative data.

- **Model 2: Siting Model**  
  Within prioritized zones, predicts the best locations for specific structures using **geospatial features** (terrain, hydrology, rainfall, LULC, vegetation indices) and **ML / hybrid ML-DL** models.

## What’s in this repository
- `midsem_report.tex` — Mid-semester report (proposal + progress + plan)
- `code/gee_pipeline.js` — Google Earth Engine pipeline to build a multi-layer composite and export patch TFRecords (prototype labels)
- `code/checkTFRecord.py` — Inspect exported TFRecord schema and feature shapes
- `code/mergeAndCrop.py` — Merge TFRecord shards and crop patches for training

## Notes
- Current Model 2 labels are **proxy NDVI-threshold labels** used to validate the pipeline; they will be replaced with **verified ground-truth** (well/check-dam performance) in the next phase.
- Study region is kept **generic** for now and will be finalized based on data availability.

## Link
- Repository: https://github.com/rahuln-07/SOLVE-Project
