
# Acoustic Sim Toolkit

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/eduds08/acoustic-sim-toolkit/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/eduds08/acoustic-sim-toolkit)](https://github.com/eduds08/acoustic-sim-toolkit/issues)
[![GitHub stars](https://img.shields.io/github/stars/eduds08/acoustic-sim-toolkit)](https://github.com/eduds08/acoustic-sim-toolkit/stargazers)

## Description

**Acoustic Sim Toolkit** is a tool for simulating acoustic waves in 2D grids, using the WebGPU API to ensure fast and efficient simulations. The tool allows editing various simulation parameters and performing different types of simulations, including reflector reconstruction using Time Reversal and Reverse Time Migration.

## Features

- **Simulate an acoustic wave without reflector in the ROI (Region of Interest)**
- **Simulate an acoustic wave with a punctual reflector in the ROI**
- **Simulate an acoustic wave with a linear reflector in the ROI**
- **Reconstruct reflector using Time Reversal**
- **Reconstruct reflector using Reverse Time Migration**

## Simulation Parameters

You can edit the following simulation parameters:
- Grid size
- Wave velocity
- Simulation time
- `dz`, `dx`, `dt`
- Position and number of receivers
- Position and number of reflectors
- Source position
- Other adjustments

## Installation

To install **Acoustic Sim Toolkit**, clone the repository and install the dependencies:

```bash
git clone https://github.com/eduds08/acoustic-sim-toolkit.git
cd acoustic-sim-toolkit
pip install -r requirements.txt
```

## Usage

To use the tool, edit the simulation parameters in the configuration file and run the main script:

```bash
python main.py
```

## Contribution

Contributions are welcome! If you want to contribute to the project, please follow these steps:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/eduds08/acoustic-sim-toolkit/blob/main/LICENSE) file for more details.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me.

---

Made by [eduds08](https://github.com/eduds08)
