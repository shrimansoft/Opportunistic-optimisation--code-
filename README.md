# Warehouse Optimization Simulation

This project implements an opportunistic optimization system for warehouse robotics using reinforcement learning. The simulation models a warehouse environment with autonomous robots, shelving systems, and order fulfillment processes.

## Features

- **Warehouse Simulation**: Complete warehouse environment with 400 shelves arranged in a 20x20 grid
- **Multi-Robot System**: Support for multiple autonomous robots with different operational modes
- **Order Management**: Dynamic order generation and fulfillment tracking
- **Gymnasium Environment**: Custom OpenAI Gym environment for reinforcement learning
- **Visualization**: Real-time visualization of warehouse operations and robot movements
- **Performance Metrics**: Order delay analysis and optimization metrics

## Installation

This project uses Poetry for dependency management. Make sure you have Poetry installed on your system.

### Prerequisites

- Python 3.8 or higher
- Poetry (install from https://python-poetry.org/docs/#installation)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Opportunistic-optimisation--code-
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

### Running the Simulation

```bash
# Using Poetry
poetry run warehouse-sim

# Or if you're in the virtual environment
python main.py
```

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
# Format code with Black
poetry run black .

# Run linting
poetry run flake8 warehouse_sim/

# Type checking
poetry run mypy warehouse_sim/
```

## Project Structure

```
warehouse_sim/
├── __init__.py
├── warehouse.py          # Main warehouse simulation logic
├── robot.py             # Robot behavior and movement
├── order.py             # Order management system
├── picking_station.py   # Picking station implementation
├── environment.py       # Gymnasium environment wrapper
└── utils.py            # Utility functions

tests/                   # Test suite
examples/               # Example usage scripts
data/                   # Generated simulation data
├── frames/            # Visualization frames
└── videos/            # Generated videos
```

## Warehouse Parameters

| Parameter | Value |
|-----------|-------|
| Number of Shelves | 400 |
| Items per Shelf | 6 |
| Warehouse Layout | 20x20 grid |
| Item Types | 50 |
| Items per Type | 48 |
| Robot Speed | 1 m/s |
| Picking Stations | 1 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License.
