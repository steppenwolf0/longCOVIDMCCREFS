import pandas as pd

## Load data
expr = pd.read_csv("GSE226260_AdditionalSamples.rawCounts.csv")

## Your gene list
genes = [
    "ENSG00000104695",
    "ENSG00000118520",
    "ENSG00000127884",
    "ENSG00000160712",
    "ENSG00000180340",
    "ENSG00000184557",
    "ENSG00000211697",
    "ENSG00000211699"
]

## Filter rows
filtered = expr[expr.iloc[:, 0].isin(genes)]

## Reorder according to gene list
filtered = filtered.set_index(expr.columns[0]).loc[genes].reset_index()

## Save result
filtered.to_csv("ExprData_filtered_ordered_subset.csv", index=False)