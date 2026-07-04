# An Integrative Systems Microbiology Algorithm for Reproducible Multi-Dataset Omics Analysis with Application to Long COVID

## Installation

The code runs with scikit-learn and a few necessary packages

```bash
pip install -r requirements.txt
```

## Datasets

- [GSE275334](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275334) (in data folder) "Immune Exhaustion in ME/CFS and long COVID" 
- [E-ENAD-46](https://www.ebi.ac.uk/gxa/experiments/E-ENAD-46/Downloads) "Lung and colon transcriptome profiling of fatal COVID-19 cases"
- [GSE157103](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157103) "Largescale Multi-omic Analysis of COVID19 Severity"
- [GSE226260](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226260) "Persistent Activation of Chronic Inflammatory Pathways in Long Covid"
- [GSE270045](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270045) "Upregulation of olfactory receptors and neuronal-associated genes highlights complex immune and neuronal dysregulation in Long COVID patients"

### Run

```bash
cd ./src
python aBioInf100.py
python summaryMulti.py
python sumFig.py
```
