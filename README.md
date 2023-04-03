# PIML
## Coarse-Grained Modeling of the SARS-CoV-2 Spike Glycoprotein by Physics-Informed Machine Learning

Liang, D., Zhang, Z., Rafailovich, M., Simon, M., Deng, Y., & Zhang, P. (2023). Coarse-Grained Modeling of the SARS-CoV-2 Spike Glycoprotein by Physics-Informed Machine Learning. Computation, 11(2), 24.



### aa_ref.pdb:
The PDB describes the all-atom spike-protein structure, and each atom’s beta value is the corresponding coarse-grain (CG) bead index that it maps to. The CG bead coordinate is the center of mass of its corresponding atom group.

The PDB file describes the all-atom spike-protein structure, and each atom’s beta value is the corresponding CG bead index that it maps to. The CG bead coordinate is the center of mass of its corresponding atom group. 

```
CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A  27      62.804 143.109 118.635  1.00  8.00      AP1  N
ATOM      2  CA  ALA A  27      63.062 143.313 117.199  1.00  8.00      AP1  C
ATOM      3  CB  ALA A  27      63.587 144.763 116.923  1.00  8.00      AP1  C
ATOM      4  C   ALA A  27      64.026 142.243 116.672  1.00  8.00      AP1  C
ATOM      5  O   ALA A  27      64.728 141.589 117.319  1.00  8.00      AP1  O
```

Atom indices and type are listed columns 2 and 3 respectively. Their corresponding beta value, or, in this case, CG bead, is listed in column 11. Each atom’s corresponding residue is located in column 4.

### cg_sample.pdb:
The PDB file describes the mapped CG protein structure from the aa_ref.pdb. Each bead coordinate is the center of mass of its respective atom group in aa_ref.pdb.

### cg-nocharge.psf
The PSF structure is the protein structure file for the CG model, containing all atom-specific information and bonds/angles/dihedrals.

### PIML_model.py
Contains the physics-informed machine learning layers/model script.

### nn.par
CHARMM forcefield parameter file contains all of the learned force-field parameters from the PIML methods that are needed to evaluate forces and energies.

### cg-min.conf
NAMD configuration file to run minimization and equilibration of CG molecular dynamics simulation (CGMD).

### cg-nvt.conf
NAMD configuration file to run production run of CGMD.
