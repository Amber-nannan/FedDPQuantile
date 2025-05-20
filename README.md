This Python code implements the proposed federated quantile inference method under LDP.

Please run 'run.sh' to obtain the simulation results.

## File Description

The directory structure for FedDPQuantile is as follows:

```{bash}
FedDPQuantile/
├── DPQuantile.py 
├── FedDPQuantile.py 
├── README.md
├── case_global_hete.py
├── case_global_hete_d.py
├── case_global_homo.py
├── case_hete.py
├── case_hete_d.py
├── case_homo.py
├── run.sh
├── util.py  # Utility functions for data generation, distribution, and result analysis.
└── util_fdp.py  # Federated utility functions including training procedures, pickle saving/loading, and experiment runners.
```

### Core Implementation Files

- **DPQuantile.py**: Base class for differentially private quantile estimation. 
- **FedDPQuantile.py**: Extends `DPQuantile` to support federated differentially private quantile estimation.
- **util.py**: Utility functions for data generation, distribution, and result analysis.
- **util_fdp.py**: Federated utility functions, including training procedures, pickle-based saving/loading, and experiment runners. 

### Experiment Cases

- **case_homo.py**: Simulation under homogeneous distribution settings, where clients share the same distribution.  
- **case_hete.py**: Simulation under heterogeneous distribution settings, where clients share the same distribution family but with different locations.  
- **case_hete_d.py**: Simulation under heterogeneous settings, where clients have different distribution families.  
- **case_global_homo.py**: Global training version for homogeneous distribution settings.  
- **case_global_hete.py**: Global training version for heterogeneous distribution settings with different locations.  
- **case_global_hete_d.py**: Global training version for fully heterogeneous settings with different distribution families.

### Execution

- **run.sh**: Shell script to execute all experiments.


## Getting Started

### Installation

This project requires the following Python libraries:
```bash
pip install numpy scipy ray
```

### Quick Start

1. Run a homogeneous distribution experiment:

```bash
python case_homo.py
```

2. Run a heterogeneous distribution experiment (same distribution family with different locations):
```bash
python case_hete.py
```

3. Run a fully heterogeneous experiment (different distribution families):
```bash
python case_hete_d.py
```

4. Run global training versions:
```bash
python case_global_homo.py
python case_global_hete.py
python case_global_hete_d.py
```

5. Run all experiments using the shell script:

```bash
chmod +x run.sh
./run.sh
```

Note : Each experiment may take several hours to complete. The shell script runs all experiments sequentially. Results will be saved in the output directory as pickle files by default.