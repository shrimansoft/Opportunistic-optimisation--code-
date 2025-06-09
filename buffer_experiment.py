#!/usr/bin/env python3
"""
Comprehensive Buffer vs No-Buffer Experiment

This script conducts a thorough experimental analysis comparing warehouse performance
with and without picking station buffers. The experiment includes:

1. Buffer enabled vs disabled comparisons
2. Variation across 5 random seeds for statistical robustness
3. Order completion time distribution analysis
4. Delay statistics (mean, variance, min, max)
5. Performance variation based on number of robots and buffer sizes
6. Statistical analysis and visualization

Author: Warehouse Optimization Team
Date: 2024
"""

import os
import sys
import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import seaborn as sns

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from warehouse_sim.warehouse import Warehouse
from warehouse_sim.order import OrderItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the buffer experiment."""

    seeds: List[int]
    simulation_steps: int
    robot_counts: List[int]
    buffer_sizes: List[int]
    buffer_enabled_scenarios: List[bool]
    output_dir: str

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationMetrics:
    """Metrics collected from a single simulation run."""

    seed: int
    buffer_enabled: bool
    robot_count: int
    buffer_size: int
    simulation_steps: int

    # Order completion metrics
    total_orders_completed: int
    total_orders_pending: int
    completion_times: List[int]
    delays: List[int]

    # Delay statistics
    mean_delay: float
    variance_delay: float
    std_delay: float
    min_delay: int
    max_delay: int
    median_delay: float

    # Performance metrics
    throughput: float  # orders completed per time step
    utilization: float  # percentage of time robots were busy

    # Buffer-specific metrics (only when buffer enabled)
    buffer_hits: int  # orders fulfilled directly from buffer
    buffer_hit_rate: float

    # Additional metrics
    final_stock: int
    average_robot_distance: float


class WarehouseExperiment:
    """Main experiment class for buffer vs no-buffer analysis."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[SimulationMetrics] = []

    def run_single_simulation(
        self, seed: int, buffer_enabled: bool, robot_count: int, buffer_size: int
    ) -> SimulationMetrics:
        """Run a single simulation with specified parameters."""

        logger.info(
            f"Running simulation: seed={seed}, buffer={buffer_enabled}, "
            f"robots={robot_count}, buffer_size={buffer_size}"
        )

        # Initialize warehouse with specific parameters
        warehouse = Warehouse(seed=seed, buffer_enabled=buffer_enabled)

        # Adjust robot count
        if robot_count != len(warehouse.robots):
            # Create new robot list with desired count
            from warehouse_sim.robot import Robot

            warehouse.robots = [Robot(warehouse, i + 1) for i in range(robot_count)]

        # Adjust buffer sizes if buffers are enabled
        if buffer_enabled:
            for station in warehouse.picking_stations:
                station.default_buffer_size = buffer_size
                station.buffer_size = buffer_size

        # Track metrics during simulation
        buffer_hits = 0
        robot_busy_time = defaultdict(int)
        robot_total_distance = defaultdict(float)

        # Run simulation
        for step in range(self.config.simulation_steps):
            # Store initial order count for buffer hit tracking
            initial_completed = len(warehouse.order_compleated)

            # Execute one simulation step
            warehouse.order_step()

            # Track buffer hits (orders completed immediately from buffer)
            if buffer_enabled:
                new_completed = len(warehouse.order_compleated)
                if new_completed > initial_completed:
                    # Check if any recent orders were completed immediately
                    recent_orders = warehouse.order_compleated[initial_completed:]
                    for order in recent_orders:
                        if order.delay == 0:  # Completed immediately from buffer
                            buffer_hits += 1

            # Update robot states and track metrics
            for robot in warehouse.robots:
                if not robot.available:
                    robot_busy_time[robot.robot_id] += 1
                    if robot.shelf_location:
                        # Calculate distance moved (simplified)
                        robot_total_distance[robot.robot_id] += 1

                robot.step()

            # Assign robots to pending orders
            warehouse.robot_assigner()

            # Progress logging
            if step % 100 == 0 and step > 0:
                avg_delay = warehouse.average_delay()
                logger.debug(
                    f"Step {step}: {len(warehouse.order_compleated)} completed, "
                    f"avg delay: {avg_delay:.2f}"
                )

        # Calculate final metrics
        metrics = self._calculate_metrics(
            warehouse,
            seed,
            buffer_enabled,
            robot_count,
            buffer_size,
            buffer_hits,
            robot_busy_time,
            robot_total_distance,
        )

        return metrics

    def _calculate_metrics(
        self,
        warehouse: Warehouse,
        seed: int,
        buffer_enabled: bool,
        robot_count: int,
        buffer_size: int,
        buffer_hits: int,
        robot_busy_time: Dict[int, int],
        robot_total_distance: Dict[int, float],
    ) -> SimulationMetrics:
        """Calculate comprehensive metrics from simulation results."""

        # Extract order completion data
        completed_orders = warehouse.order_compleated
        completion_times = [
            order.done_time for order in completed_orders if order.done_time is not None
        ]
        delays = [order.delay for order in completed_orders]

        # Handle case with no completed orders
        if not delays:
            delays = [0]
            completion_times = [0]

        # Calculate delay statistics
        delays_array = np.array(delays)
        mean_delay = float(np.mean(delays_array))
        variance_delay = float(np.var(delays_array))
        std_delay = float(np.std(delays_array))
        min_delay = int(np.min(delays_array))
        max_delay = int(np.max(delays_array))
        median_delay = float(np.median(delays_array))

        # Calculate performance metrics
        total_completed = len(completed_orders)
        total_pending = len(warehouse.order_buffer)
        throughput = total_completed / self.config.simulation_steps

        # Calculate robot utilization
        total_robot_time = robot_count * self.config.simulation_steps
        total_busy_time = sum(robot_busy_time.values())
        utilization = total_busy_time / total_robot_time if total_robot_time > 0 else 0

        # Calculate buffer metrics
        buffer_hit_rate = buffer_hits / total_completed if total_completed > 0 else 0

        # Calculate average robot distance
        avg_robot_distance = (
            sum(robot_total_distance.values()) / len(robot_total_distance)
            if robot_total_distance
            else 0
        )

        return SimulationMetrics(
            seed=seed,
            buffer_enabled=buffer_enabled,
            robot_count=robot_count,
            buffer_size=buffer_size,
            simulation_steps=self.config.simulation_steps,
            total_orders_completed=total_completed,
            total_orders_pending=total_pending,
            completion_times=completion_times,
            delays=delays,
            mean_delay=mean_delay,
            variance_delay=variance_delay,
            std_delay=std_delay,
            min_delay=min_delay,
            max_delay=max_delay,
            median_delay=median_delay,
            throughput=throughput,
            utilization=utilization,
            buffer_hits=buffer_hits,
            buffer_hit_rate=buffer_hit_rate,
            final_stock=int(warehouse.stock.sum()),
            average_robot_distance=avg_robot_distance,
        )

    def run_experiment(self):
        """Run the complete experiment across all parameter combinations."""

        logger.info("Starting comprehensive buffer experiment...")
        start_time = time.time()

        total_runs = (
            len(self.config.seeds)
            * len(self.config.buffer_enabled_scenarios)
            * len(self.config.robot_counts)
            * len(self.config.buffer_sizes)
        )

        logger.info(f"Total simulation runs planned: {total_runs}")

        run_count = 0

        for seed in self.config.seeds:
            for buffer_enabled in self.config.buffer_enabled_scenarios:
                for robot_count in self.config.robot_counts:
                    for buffer_size in self.config.buffer_sizes:
                        # Skip buffer_size variations when buffer is disabled
                        if (
                            not buffer_enabled
                            and buffer_size != self.config.buffer_sizes[0]
                        ):
                            continue

                        run_count += 1
                        logger.info(f"Run {run_count}/{total_runs}")

                        try:
                            metrics = self.run_single_simulation(
                                seed, buffer_enabled, robot_count, buffer_size
                            )
                            self.results.append(metrics)

                        except Exception as e:
                            logger.error(f"Error in simulation run {run_count}: {e}")
                            continue

        elapsed_time = time.time() - start_time
        logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully completed {len(self.results)} simulations")

        # Save results
        self.save_results()

    def save_results(self):
        """Save experiment results to files."""

        # Save raw data as pickle
        pickle_path = Path(self.config.output_dir) / "experiment_results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.results, f)

        # Save results as JSON (for readability)
        json_path = Path(self.config.output_dir) / "experiment_results.json"
        json_data = [asdict(result) for result in self.results]
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Results saved to {pickle_path} and {json_path}")

    def load_results(self, filepath: str = None):
        """Load experiment results from file."""
        if filepath is None:
            filepath = Path(self.config.output_dir) / "experiment_results.pkl"

        with open(filepath, "rb") as f:
            self.results = pickle.load(f)

        logger.info(f"Loaded {len(self.results)} results from {filepath}")


class ExperimentAnalyzer:
    """Analyzer for experiment results with statistical tests and visualizations."""

    def __init__(self, results: List[SimulationMetrics], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def analyze_buffer_impact(self):
        """Analyze the impact of buffer enabled vs disabled."""

        logger.info("Analyzing buffer impact...")

        # Separate results by buffer status
        buffer_enabled = [r for r in self.results if r.buffer_enabled]
        buffer_disabled = [r for r in self.results if not r.buffer_enabled]

        if not buffer_enabled or not buffer_disabled:
            logger.warning("Missing data for buffer comparison")
            return

        # Statistical comparison
        metrics_to_compare = ["mean_delay", "throughput", "utilization"]

        results_summary = {}

        for metric in metrics_to_compare:
            enabled_values = [getattr(r, metric) for r in buffer_enabled]
            disabled_values = [getattr(r, metric) for r in buffer_disabled]

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(enabled_values, disabled_values)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((np.std(enabled_values) ** 2 + np.std(disabled_values) ** 2) / 2)
            )
            cohens_d = (np.mean(enabled_values) - np.mean(disabled_values)) / pooled_std

            results_summary[metric] = {
                "buffer_enabled_mean": np.mean(enabled_values),
                "buffer_enabled_std": np.std(enabled_values),
                "buffer_disabled_mean": np.mean(disabled_values),
                "buffer_disabled_std": np.std(disabled_values),
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "significant": p_value < 0.05,
            }

        # Save statistical analysis
        stats_path = self.output_dir / "buffer_impact_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        logger.info(f"Statistical analysis saved to {stats_path}")

        # Create visualizations
        self._plot_buffer_comparison(buffer_enabled, buffer_disabled)

    def analyze_delay_distributions(self):
        """Analyze order completion time distributions."""

        logger.info("Analyzing delay distributions...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Order Delay Distribution Analysis", fontsize=16, fontweight="bold"
        )

        # Group results by buffer status
        buffer_groups = {
            "Buffer Enabled": [r for r in self.results if r.buffer_enabled],
            "Buffer Disabled": [r for r in self.results if not r.buffer_enabled],
        }

        colors = ["#2E86AB", "#A23B72"]

        # Plot 1: Delay distribution histograms
        ax1 = axes[0, 0]
        for i, (label, group) in enumerate(buffer_groups.items()):
            all_delays = []
            for result in group:
                all_delays.extend(result.delays)

            if all_delays:
                ax1.hist(
                    all_delays,
                    bins=30,
                    alpha=0.7,
                    label=label,
                    color=colors[i],
                    density=True,
                )

        ax1.set_xlabel("Order Delay (time steps)")
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution of Order Delays")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Box plot of mean delays by configuration
        ax2 = axes[0, 1]
        delay_data = []
        labels = []

        for buffer_enabled in [True, False]:
            for robot_count in sorted(set(r.robot_count for r in self.results)):
                group_delays = [
                    r.mean_delay
                    for r in self.results
                    if r.buffer_enabled == buffer_enabled
                    and r.robot_count == robot_count
                ]
                if group_delays:
                    delay_data.append(group_delays)
                    labels.append(
                        f"{'Buffer' if buffer_enabled else 'No Buffer'}\n{robot_count} robots"
                    )

        if delay_data:
            bp = ax2.boxplot(delay_data, tick_labels=labels, patch_artist=True)

            # Color boxes
            for i, patch in enumerate(bp["boxes"]):
                color = (
                    colors[0]
                    if "Buffer" in labels[i] and "No Buffer" not in labels[i]
                    else colors[1]
                )
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_ylabel("Mean Delay (time steps)")
        ax2.set_title("Mean Delay by Robot Count and Buffer Status")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Throughput comparison
        ax3 = axes[1, 0]
        throughput_data = []
        throughput_labels = []

        for buffer_enabled in [True, False]:
            group_throughput = [
                r.throughput for r in self.results if r.buffer_enabled == buffer_enabled
            ]
            if group_throughput:
                throughput_data.append(group_throughput)
                throughput_labels.append(
                    "Buffer Enabled" if buffer_enabled else "Buffer Disabled"
                )

        if throughput_data:
            bp = ax3.boxplot(
                throughput_data, tick_labels=throughput_labels, patch_artist=True
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)

        ax3.set_ylabel("Throughput (orders/timestep)")
        ax3.set_title("System Throughput Comparison")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Delay variance comparison
        ax4 = axes[1, 1]
        variance_data = []
        variance_labels = []

        for buffer_enabled in [True, False]:
            group_variance = [
                r.variance_delay
                for r in self.results
                if r.buffer_enabled == buffer_enabled
            ]
            if group_variance:
                variance_data.append(group_variance)
                variance_labels.append(
                    "Buffer Enabled" if buffer_enabled else "Buffer Disabled"
                )

        if variance_data:
            bp = ax4.boxplot(
                variance_data, tick_labels=variance_labels, patch_artist=True
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)

        ax4.set_ylabel("Delay Variance")
        ax4.set_title("Delay Variance Comparison")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "delay_distribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Delay distribution analysis saved to {plot_path}")

    def analyze_robot_buffer_variations(self):
        """Analyze performance variations based on robot count and buffer size."""

        logger.info("Analyzing robot count and buffer size variations...")

        # Create heatmaps for different metrics
        robot_counts = sorted(set(r.robot_count for r in self.results))
        buffer_sizes = sorted(
            set(r.buffer_size for r in self.results if r.buffer_enabled)
        )

        if not buffer_sizes:
            logger.warning("No buffer size variations found")
            return

        metrics = ["mean_delay", "throughput", "utilization", "buffer_hit_rate"]
        metric_titles = [
            "Mean Delay",
            "Throughput",
            "Robot Utilization",
            "Buffer Hit Rate",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Performance Heatmaps: Robot Count vs Buffer Size",
            fontsize=16,
            fontweight="bold",
        )

        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]

            # Create matrix for heatmap
            heatmap_data = np.zeros((len(robot_counts), len(buffer_sizes)))

            for i, robot_count in enumerate(robot_counts):
                for j, buffer_size in enumerate(buffer_sizes):
                    # Get results matching this configuration (buffer enabled only)
                    matching_results = [
                        r
                        for r in self.results
                        if (
                            r.robot_count == robot_count
                            and r.buffer_size == buffer_size
                            and r.buffer_enabled
                        )
                    ]

                    if matching_results:
                        values = [getattr(r, metric) for r in matching_results]
                        heatmap_data[i, j] = np.mean(values)
                    else:
                        heatmap_data[i, j] = np.nan

            # Create heatmap
            im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")

            # Set ticks and labels
            ax.set_xticks(range(len(buffer_sizes)))
            ax.set_yticks(range(len(robot_counts)))
            ax.set_xticklabels(buffer_sizes)
            ax.set_yticklabels(robot_counts)

            ax.set_xlabel("Buffer Size")
            ax.set_ylabel("Robot Count")
            ax.set_title(title)

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Add text annotations
            for i in range(len(robot_counts)):
                for j in range(len(buffer_sizes)):
                    if not np.isnan(heatmap_data[i, j]):
                        text = f"{heatmap_data[i, j]:.2f}"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color="white"
                            if heatmap_data[i, j] < np.nanmean(heatmap_data)
                            else "black",
                        )

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "robot_buffer_heatmaps.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Robot/buffer variation analysis saved to {plot_path}")

    def _plot_buffer_comparison(self, buffer_enabled: List, buffer_disabled: List):
        """Create comparison plots for buffer enabled vs disabled."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Buffer Enabled vs Disabled Performance Comparison",
            fontsize=16,
            fontweight="bold",
        )

        # Define metrics to plot
        metrics_info = [
            ("mean_delay", "Mean Delay (timesteps)", "lower_better"),
            ("throughput", "Throughput (orders/timestep)", "higher_better"),
            ("utilization", "Robot Utilization (%)", "higher_better"),
            ("max_delay", "Maximum Delay (timesteps)", "lower_better"),
            ("std_delay", "Delay Standard Deviation", "lower_better"),
            ("buffer_hit_rate", "Buffer Hit Rate (%)", "higher_better"),
        ]

        for idx, (metric, ylabel, preference) in enumerate(metrics_info):
            ax = axes[idx // 3, idx % 3]

            # Extract data
            if metric == "buffer_hit_rate":
                enabled_values = [getattr(r, metric) * 100 for r in buffer_enabled]
                disabled_values = [0] * len(
                    buffer_disabled
                )  # No buffer hits when disabled
            elif metric == "utilization":
                enabled_values = [getattr(r, metric) * 100 for r in buffer_enabled]
                disabled_values = [getattr(r, metric) * 100 for r in buffer_disabled]
            else:
                enabled_values = [getattr(r, metric) for r in buffer_enabled]
                disabled_values = [getattr(r, metric) for r in buffer_disabled]

            # Create box plots
            data = [enabled_values, disabled_values]
            labels = ["Buffer Enabled", "Buffer Disabled"]

            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

            # Color based on preference
            colors = ["#2E86AB", "#A23B72"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} Comparison")
            ax.grid(True, alpha=0.3)

            # Add statistical significance marker
            if len(enabled_values) > 1 and len(disabled_values) > 1:
                t_stat, p_value = stats.ttest_ind(enabled_values, disabled_values)
                if p_value < 0.05:
                    ax.text(
                        0.5,
                        0.95,
                        f"p < 0.05*",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontweight="bold",
                        color="red",
                    )

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "buffer_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Buffer comparison plots saved to {plot_path}")

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""

        logger.info("Generating summary report...")

        report_path = self.output_dir / "experiment_summary_report.txt"

        with open(report_path, "w") as f:
            f.write("WAREHOUSE BUFFER EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Experiment overview
            f.write(f"Total Simulations Run: {len(self.results)}\n")
            f.write(f"Seeds Used: {sorted(set(r.seed for r in self.results))}\n")
            f.write(
                f"Robot Counts Tested: {sorted(set(r.robot_count for r in self.results))}\n"
            )
            f.write(
                f"Buffer Sizes Tested: {sorted(set(r.buffer_size for r in self.results))}\n"
            )
            f.write(
                f"Simulation Steps per Run: {self.results[0].simulation_steps if self.results else 'N/A'}\n\n"
            )

            # Buffer comparison summary
            buffer_enabled = [r for r in self.results if r.buffer_enabled]
            buffer_disabled = [r for r in self.results if not r.buffer_enabled]

            if buffer_enabled and buffer_disabled:
                f.write("BUFFER IMPACT SUMMARY\n")
                f.write("-" * 25 + "\n")

                # Calculate averages
                metrics = ["mean_delay", "throughput", "utilization", "max_delay"]

                for metric in metrics:
                    enabled_avg = np.mean([getattr(r, metric) for r in buffer_enabled])
                    disabled_avg = np.mean(
                        [getattr(r, metric) for r in buffer_disabled]
                    )
                    improvement = (
                        (enabled_avg - disabled_avg) / disabled_avg * 100
                        if disabled_avg != 0
                        else 0
                    )

                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Buffer Enabled:  {enabled_avg:.3f}\n")
                    f.write(f"  Buffer Disabled: {disabled_avg:.3f}\n")
                    f.write(f"  Improvement:     {improvement:+.1f}%\n\n")

                # Buffer hit statistics
                total_buffer_hits = sum(r.buffer_hits for r in buffer_enabled)
                total_completed = sum(r.total_orders_completed for r in buffer_enabled)
                overall_hit_rate = (
                    total_buffer_hits / total_completed * 100
                    if total_completed > 0
                    else 0
                )

                f.write(f"Buffer Hit Rate: {overall_hit_rate:.1f}%\n")
                f.write(f"Total Buffer Hits: {total_buffer_hits}\n\n")

            # Best configurations
            f.write("BEST PERFORMING CONFIGURATIONS\n")
            f.write("-" * 35 + "\n")

            # Best for minimum delay
            best_delay = min(self.results, key=lambda r: r.mean_delay)
            f.write(f"Minimum Mean Delay: {best_delay.mean_delay:.2f} timesteps\n")
            f.write(
                f"  Configuration: {best_delay.robot_count} robots, "
                f"buffer {'enabled' if best_delay.buffer_enabled else 'disabled'}"
            )
            if best_delay.buffer_enabled:
                f.write(f", buffer size {best_delay.buffer_size}")
            f.write(f", seed {best_delay.seed}\n\n")

            # Best for throughput
            best_throughput = max(self.results, key=lambda r: r.throughput)
            f.write(
                f"Maximum Throughput: {best_throughput.throughput:.4f} orders/timestep\n"
            )
            f.write(
                f"  Configuration: {best_throughput.robot_count} robots, "
                f"buffer {'enabled' if best_throughput.buffer_enabled else 'disabled'}"
            )
            if best_throughput.buffer_enabled:
                f.write(f", buffer size {best_throughput.buffer_size}")
            f.write(f", seed {best_throughput.seed}\n\n")

            # Statistical significance tests
            f.write("STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-" * 35 + "\n")

            if buffer_enabled and buffer_disabled:
                for metric in ["mean_delay", "throughput", "utilization"]:
                    enabled_values = [getattr(r, metric) for r in buffer_enabled]
                    disabled_values = [getattr(r, metric) for r in buffer_disabled]

                    t_stat, p_value = stats.ttest_ind(enabled_values, disabled_values)
                    significance = "Yes" if p_value < 0.05 else "No"

                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  t-statistic: {t_stat:.3f}\n")
                    f.write(f"  p-value:     {p_value:.6f}\n")
                    f.write(f"  Significant: {significance} (α = 0.05)\n\n")

        logger.info(f"Summary report saved to {report_path}")


def main():
    """Main function to run the comprehensive buffer experiment."""

    # Define experiment configuration
    config = ExperimentConfig(
        seeds=[42, 123, 456, 789, 999],
        simulation_steps=5000,
        robot_counts=[3, 4, 5, 6, 7],
        buffer_sizes=[4, 6, 8, 10, 12],
        buffer_enabled_scenarios=[True, False],
        output_dir="experiment_results_10",
    )

    logger.info("Starting comprehensive warehouse buffer experiment")
    logger.info(f"Configuration: {asdict(config)}")

    # Run experiment
    experiment = WarehouseExperiment(config)
    experiment.run_experiment()

    # Analyze results
    analyzer = ExperimentAnalyzer(experiment.results, config.output_dir)
    analyzer.analyze_buffer_impact()
    analyzer.analyze_delay_distributions()
    analyzer.analyze_robot_buffer_variations()
    analyzer.generate_summary_report()

    logger.info("Experiment completed successfully!")
    logger.info(f"Results available in: {config.output_dir}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("EXPERIMENT QUICK SUMMARY")
    print("=" * 60)

    buffer_enabled = [r for r in experiment.results if r.buffer_enabled]
    buffer_disabled = [r for r in experiment.results if not r.buffer_enabled]

    if buffer_enabled and buffer_disabled:
        enabled_delay = np.mean([r.mean_delay for r in buffer_enabled])
        disabled_delay = np.mean([r.mean_delay for r in buffer_disabled])
        delay_improvement = (
            (disabled_delay - enabled_delay) / disabled_delay * 100
            if disabled_delay > 0
            else 0
        )

        enabled_throughput = np.mean([r.throughput for r in buffer_enabled])
        disabled_throughput = np.mean([r.throughput for r in buffer_disabled])
        throughput_improvement = (
            (enabled_throughput - disabled_throughput) / disabled_throughput * 100
            if disabled_throughput > 0
            else 0
        )

        print(f"Average delay improvement with buffers: {delay_improvement:+.1f}%")
        print(
            f"Average throughput improvement with buffers: {throughput_improvement:+.1f}%"
        )

        total_buffer_hits = sum(r.buffer_hits for r in buffer_enabled)
        total_completed = sum(r.total_orders_completed for r in buffer_enabled)
        hit_rate = (
            total_buffer_hits / total_completed * 100 if total_completed > 0 else 0
        )

        print(f"Overall buffer hit rate: {hit_rate:.1f}%")
        print(f"Total simulations completed: {len(experiment.results)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
