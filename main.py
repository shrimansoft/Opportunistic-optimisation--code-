"""
Main entry point for the RWare-style Warehouse Simulation Demo.

This module serves as the primary entry point for running warehouse simulation
demonstrations. It orchestrates the various components of the simulation system
including environment setup, demo execution, and visualization.
"""

from simulation import DemoRunner, SimulationConfig


def main() -> None:
    """
    Main function for running the warehouse simulation demo.
    
    This function initializes the demo configuration and runs the RWare-style
    warehouse demonstration across different layout sizes.
    """
    # Create simulation configuration
    # You can customize this to use different configurations:
    # - SimulationConfig.quick_demo() for fast demos
    # - SimulationConfig.comprehensive_demo() for full demos
    # - SimulationConfig.default() for standard demos
    config = SimulationConfig.default()
    
    # Initialize and run the demo
    demo_runner = DemoRunner(config)
    demo_runner.run_rware_demo()


if __name__ == "__main__":
    main()

