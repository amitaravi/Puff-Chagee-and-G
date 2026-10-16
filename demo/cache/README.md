# Showcase cache

This directory contains repository-backed Config C verdicts and analyses for the four
curated demo use cases. The Streamlit app loads these files in a fresh session so the
Case Analysis tab is immediately available without rerunning the tribunal or calling a
model.

Each JSON file contains only case output already intended for presentation: the verdict,
both advocate briefs, and the generated analysis. Evidence packs remain sourced from
`data/cases/`.
