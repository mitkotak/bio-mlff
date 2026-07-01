# Bio-MLFF

Bio-MLFF contains JAX model implementations that run through [Openmm-JAX's JaxForce](https://github.com/mitkotak/openmm-jax)

```bash
pip install bio-mlff
```

```python
import biomlff.anipotential
from openmmml import MLPotential

potential = MLPotential("ani2x-jax-model0")
```

## Supported Models

Except FeNNix, all the models are implemented from scratch with JAX-MD and Equinox as the only dependencies.

- `fennix-bio1-small`, `fennix-bio1-medium`
- `ani2x-jax-model0`, `ani2x-jax-ensemble` 
- `mace-jax-off-s-23`,  `mace-jax-off-m-24`
- `aimnet2-jax`
- `aceff-jax-1.1`, `aceff-jax-2.0`
- `orb-jax-v3-conservative-omol`
- `so3lr` 

# Acknowledgements

@abhijeetgangan for discussions on API design [openmm-ml](https://github.com/openmm/openmm-ml) for the tests/API, [FeNNol](https://github.com/FeNNol-tools/FeNNol) for their ANI implementation. [@teddykoker](https://github.com/atomicarchitects/nequix/blob/main/nequix/model.py) for from scratch mindset.