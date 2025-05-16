## Quantum Particle Simulation

This repository contains a set of Python scripts and accompanying plots that simulate the time evolution of a quantum particle interacting with various potential barriers. Each script implements the split-step Fourier method to solve the time-dependent Schrödinger equation in two dimensions.

## 📂 Repository Structure
```
├── potential_barriers/           # Simulation scripts for different potentials
│   ├── Checkerboard.py           # Checkerboard-patterned barrier
│   ├── Quantum Tunnel.py         # Weak barrier for quantum tunneling
│   ├── Single Slit.py            # Strong barrier with single slit
│   ├── Young's Double Slit Experiment.py  # Double-slit barrier
│   ├── no slit.py                # Strong barrier with no slits (reflector)
│   └── Tunnel.py                 # Tunnel-like barrier environment
│
├── plots/                        # Simulation output images
│   ├── chessboard/               # Checkerboard results (.png)
│   ├── double_slit/              # Double-slit results
│   ├── no_slit/                  # No-slit results
│   ├── quantum_tunneling/        # Tunneling results
│   ├── single_slit/              # Single-slit results
│   └── tunnel/                   # Tunnel environment results
│
├── 2D_Quantum_Particle_2.pdf  # Base python code all variations are based on
│── Quantum_Physics_Comp_Report_2.pdf  # Detailed report of methods and findings
|
└── README.md                     # You are here
```
## 🧩 Dependencies

- Python 3.7 or higher

- NumPy

- Matplotlib

Install dependencies via pip:
```
pip install numpy matplotlib
```
## 🚀 Running Simulations

1. Navigate to the potential_barriers/ directory:
```
cd potential_barriers
```
2. Run any script. For example, to simulate the single-slit barrier:
```
python "Single Slit.py"
```
3. Plotting: Each script includes commented-out `plt.imshow` blocks. Uncomment the plotting lines in the script to generate real-time figures, or run the script as-is to save figures programmatically.

Note: Modify the parameters (N, sigma, delta_t, t_total, etc.) at the top of each script to adjust grid size, Gaussian width, time step, and total evolution time.

## 📐 Script Overview

| Script                          | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| Checkerboard.py                 | Barrier arranged in a checkerboard pattern               |
| Quantum Tunnel.py               | Weak uniform barrier to showcase tunneling effects       |
| Single Slit.py                  | Single narrow slit in a strong barrier                   |
| Young's Double Slit Experiment.py | Two slits to recreate the classic Young's experiment       |
| no slit.py                      | Solid barrier with no openings (total reflection)        |
| Tunnel.py                       | Narrow tunnel-like opening for guided propagation        |


## 📊 Results

Generated plots are saved in the plots/ folder under subdirectories matching each script. Each .png shows either the absolute value or phase distribution of the wavefunction at specified time intervals.

## 📝 Report

See Quantum_Physics_Comp_Report_2.pdf for a detailed discussion of:

Numerical method (split-step Fourier)

Parameter choices and physical interpretation

Analysis of diffraction, interference, and tunneling phenomena

## 🛡 License

This project is released under the MIT License.

© Campbell Timms
