# Boolean Network Inference for *Clostridium beijerinckii* NRRL B-598

Genome-wide Boolean network inference from RNA-Seq (18 BAMs: B,C,D × 6 time points).  
Implements discretization, regulator prefiltering, decision-tree rule extraction → SBML-qual.

## Data
- Source: SRA SRP033480 (3 biological replicates × 6 time points = 18 BAMs).


## Quick start
```bash
python -V  # 3.10+
pip install numpy pandas scikit-learn
