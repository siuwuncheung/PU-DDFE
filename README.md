# Introduction
PU-DDFE is a Python software for Partition-of-Unity Data-Driven Finite Element Methods (PU-DDFEM), 
which operates via four core mechanisms: 
1. training localized models on small subdomains or components,
2. assembling them globally using partition-of-unity and finite-element coupling principles,
3. explicitly solving the governing equations of the physical system, and 
4. reusing pretrained components across entirely unseen global systems. 

## Features
- Partition-of-unity assembly
- Projection-based reduced order model
- Supporting physics equations:
  - Anisotropic diffusion
