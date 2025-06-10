
import matplotlib
import numpy as np

import os

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import psutil
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


import matplotlib.pyplot as plt

# Configure matplotlib for interactive plotting
plt.ion()
# Disable toolbar to prevent TclError issues
matplotlib.rcParams['toolbar'] = 'None'


def ploter(self, frame_dir=None, step_number=None, pause_time=0.1):
    """Interactive version of enhanced plot that displays in real-time."""
    if not self.interactive_mode and frame_dir is None:
        return

    if step_number is None:
        step_number = self.time

    frame = self.time

    # --- 1. Data Preparation ---
    shelf_counts = np.array([len(a) for a in self.shelfs])
    warehouse_layout = shelf_counts.reshape(20, 20)
    shelf_buffer = np.array([(i in self.itemShelfsBufferSet) for i in range(400)])
    shelf_buffer_layout = shelf_buffer.reshape(20, 20)
    total_stock = self.stock.sum()
    total_orders = len(self.order_buffer)
    completed_orders = len(self.order_compleated)

    # Create figure only once for interactive mode
    if self.fig is None:
        self.fig = plt.figure(figsize=(18, 22), facecolor="#f0f0f0")
        plt.show(block=False)  # Non-blocking show
    # Clear only the content, not the figure
    self.fig.clear()

    # --- 2. Figure and Layout Setup ---
    gs = GridSpec(4, 2, figure=self.fig, height_ratios=[8, 2.5, 7, 1], hspace=0.6, wspace=0.15)

    ax_warehouse = self.fig.add_subplot(gs[0, 0])
    ax_buffer = self.fig.add_subplot(gs[0, 1])
    ax_info = self.fig.add_subplot(gs[1, :])
    ax_monitor = self.fig.add_subplot(gs[2, :])
    ax_legend = self.fig.add_subplot(gs[3, :])

    # Set panel backgrounds to white for contrast
    for ax in [ax_warehouse, ax_buffer, ax_info, ax_monitor, ax_legend]:
        ax.set_facecolor("white")

    # --- 3. Plot Main Visuals ---
    cmap = mcolors.ListedColormap(["#ffffff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"])
    norm = mcolors.BoundaryNorm(np.arange(0, 8), cmap.N)
    ax_warehouse.imshow(warehouse_layout, cmap=cmap, norm=norm, interpolation="nearest")
    ax_buffer.imshow(shelf_buffer_layout, cmap=mcolors.ListedColormap(["#ffffff", "#08519c"]), interpolation="nearest")

    def style_main_ax(ax, title):
        ax.set_xticks(np.arange(0, 20, 2))
        ax.set_yticks(np.arange(0, 20, 2))
        ax.set_xticklabels(np.arange(1, 21, 2))
        ax.set_yticklabels(np.arange(1, 21, 2))
        ax.set_title(title, fontsize=16, weight="bold")
        ax.grid(False)

    style_main_ax(ax_warehouse, "Warehouse Shelf Distribution")
    style_main_ax(ax_buffer, "Order Buffer")

    # Plot picking stations
    for i, station in enumerate(self.picking_stations):
        y, x = station.location[1] - 1, station.location[0] - 1
        if not station.buffer_enabled or station.buffer_size == 0:
            color = "dimgray"
        else:
            ratio = len(station.buffer) / station.buffer_size
            if ratio >= 0.8:
                color = "red"
            elif ratio >= 0.5:
                color = "orange"
            else:
                color = "purple"
        for ax in [ax_warehouse, ax_buffer]:
            ax.plot(y, x, "s", ms=12, color=color, mec="black", mew=2)
            ax.text(y, x, f"{i}", color="white", fontsize=8, ha="center", va="center", weight="bold")

    # Plot robots
    for robot in self.robots:
        y, x = robot.current_location[1] - 1, robot.current_location[0] - 1
        color = {0: "green", 1: "blue", 2: "orange", 3: "red"}.get(robot.mode, "black")
        for ax in [ax_warehouse, ax_buffer]:
            ax.plot(y, x, "o", markersize=8, color=color, mec="white", mew=0.5)
            ax.text(y, x, f"{robot.robot_id}", color="white", fontsize=5, ha="center", va="center")
        if robot.shelf_location:
            sy, sx = robot.shelf_location[0] - 1, robot.shelf_location[1] - 1
            ax_buffer.plot(sx, sy, "D", markersize=8, color="#08519c")
            ax_buffer.text(sx, sy, f"{robot.robot_id}", color="white", fontsize=5, ha="center", va="center")

    # --- 4. General Information Panel ---
    ax_info.axis("off")
    # Added a subtle border to the panel
    for spine in ax_info.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("lightgrey")

    ax_info.set_title("📦 Warehouse Status", fontsize=16, weight="bold", pad=15)
    info_text = f"Total Stock: {int(total_stock):,} | Orders in Progress: {total_orders} | Completed Orders: {completed_orders}"
    ax_info.text(0.5, 0.85, info_text, ha="center", va="center", fontsize=14, weight="bold", transform=ax_info.transAxes)

    # Re-organized payload and buffer info into a more compact 3-column layout
    robot_shelf_info = [f"R{r.robot_id} → Shelf {r.shelf}" if r.shelf is not None else f"R{r.robot_id} (Idle)" for r in self.robots]
    mid_point = (len(robot_shelf_info) + 1) // 2

    payload_col1 = "Robot Payloads:\n" + "\n".join(robot_shelf_info[:mid_point])
    payload_col2 = "\n" + "\n".join(robot_shelf_info[mid_point:])

    buffer_lines = ["Picking Station Buffers:"]
    for i, station in enumerate(self.picking_stations):
        status = "DISABLED" if not station.buffer_enabled else f"{len(station.buffer)}/{station.buffer_size} items"
        buffer_lines.append(f"PS{i}: {status}")
        # Truncate long buffer lists
        if station.buffer:
            buffer_str = str(station.buffer)
            if len(buffer_str) > 25:
                buffer_str = buffer_str[:22] + "...]"
            buffer_lines.append(f"  └> {buffer_str}")
    buffer_col = "\n".join(buffer_lines)

    ax_info.text(0.05, 0.6, payload_col1, ha="left", va="top", fontsize=11, fontfamily="monospace", transform=ax_info.transAxes)
    ax_info.text(0.35, 0.6, payload_col2, ha="left", va="top", fontsize=11, fontfamily="monospace", transform=ax_info.transAxes)
    ax_info.text(0.65, 0.6, buffer_col, ha="left", va="top", fontsize=11, fontfamily="monospace", transform=ax_info.transAxes)

    # --- 5. System Monitor Panel ---
    ax_monitor.axis("off")
    ax_monitor.set_title("📊 System Monitor & Performance Metrics", fontsize=16, weight="bold", pad=20)

    # Performance calculations
    avg_delay = self.average_delay()
    throughput = completed_orders / max(self.time, 1)
    robot_util = len([r for r in self.robots if not r.available]) / len(self.robots) * 100
    total_buffer_items = sum(len(s.buffer) for s in self.picking_stations)
    total_buffer_capacity = sum(s.buffer_size for s in self.picking_stations if s.buffer_enabled)
    buffer_fill = (total_buffer_items / max(total_buffer_capacity, 1)) * 100
    buffer_hits = len([o for o in self.order_compleated if o.delay == 0])
    hit_rate = (buffer_hits / max(completed_orders, 1)) * 100

    # Left side: Warehouse performance
    robot_status_lines = []
    for r in self.robots:
        mode_map = {0: "🟢 Idle", 1: "🔵 →Shelf", 2: "🟠 →Station", 3: "🔴 →Return"}
        shelf_info = f"Shelf{r.shelf:<3}" if r.shelf else "No Shelf "
        # Added color and better text for robot time status
        if r.time_left > 0:
            time_info = f"{r.time_left:>2}t"
            time_color = "darkorange"
        else:
            time_info = "Ready"
            time_color = "green"
        status_line = f"  • R{r.robot_id}: {mode_map.get(r.mode, 'Unknown')} | {shelf_info} | "
        robot_status_lines.append((status_line, time_info, time_color))

    left_metrics_header = f"""WAREHOUSE PERFORMANCE
──────────────────────────
🔢 Step: {step_number:<7,} | ⏰ Time: {self.time:<5,}
⚡ Avg Delay: {avg_delay:.2f} steps
📈 Throughput: {throughput:.3f} ord/step
🤖 Robot Utilization: {robot_util:.1f}%
💾 Buffer Fill: {buffer_fill:.1f}% ({total_buffer_items}/{total_buffer_capacity})
🎯 Buffer Hit Rate: {hit_rate:.1f}%
"""
    ax_monitor.text(0.02, 0.98, left_metrics_header, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

    robot_header = "ROBOT STATUS\n───────────"
    ax_monitor.text(0.02, 0.45, robot_header, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

    current_y = 0.38
    for base_text, time_text, time_color in robot_status_lines:
        ax_monitor.text(0.02, current_y, base_text, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")
        ax_monitor.text(0.32, current_y, time_text, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace", color=time_color, weight="bold")
        current_y -= 0.05

    # Right side: System resources
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0

        # Major refactor to create a clean, table-like layout for system stats and progress bars
        system_header = "SYSTEM MONITOR\n──────────────────────────"
        ax_monitor.text(0.52, 0.98, system_header, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

        metrics_data = [
            ("🖥️ CPU Usage", f"{cpu:.1f}%", cpu),
            ("🧠 Memory", f"{mem.percent:.1f}%", mem.percent),
            ("💿 Disk Usage", f"{disk.percent:.1f}%", disk.percent),
        ]

        bar_h = 0.04
        for i, (label, value, percent) in enumerate(metrics_data):
            y_pos = 0.88 - i * 0.08
            # Metric Label and Value
            ax_monitor.text(0.52, y_pos, label, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")
            ax_monitor.text(0.70, y_pos, value, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace", weight="bold")

            # Progress Bar
            bar_x, bar_w = 0.78, 0.20
            color = "red" if percent > 85 else "orange" if percent > 65 else "limegreen"
            ax_monitor.add_patch(patches.Rectangle((bar_x, y_pos - 0.01), bar_w, bar_h, facecolor="#e0e0e0", transform=ax_monitor.transAxes, zorder=1))
            ax_monitor.add_patch(patches.Rectangle((bar_x, y_pos - 0.01), bar_w * (percent / 100), bar_h, facecolor=color, transform=ax_monitor.transAxes, zorder=2))

        ax_monitor.text(0.52, 0.64, f"⚖️ System Load: {load:.2f}", transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

        # Recent Performance section
        perf_header = "\n\nRECENT PERFORMANCE\n───────────────────"
        ax_monitor.text(0.52, 0.55, perf_header, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

        if completed_orders > 0:
            recent_delays = str([order.delay for order in self.order_compleated[-10:]])
            perf_text = f"• Recent Delays: {recent_delays}\n"
        else:
            perf_text = "• Recent Delays: []\n"

        if total_orders > 0 and len(self.robots) > 0:
            avg_robot_time = sum(r.time_left for r in self.robots) / len(self.robots)
            expected_delay = max(0, avg_robot_time + (total_orders / len(self.robots)) * 2)
            perf_text += f"• Expected Delay: {expected_delay:.1f} steps"
        else:
            perf_text += "• Expected Delay: 0.0 steps"

        ax_monitor.text(0.53, 0.40, perf_text, transform=ax_monitor.transAxes, fontsize=11, va="top", fontfamily="monospace")

    except Exception as e:
        ax_monitor.text(0.52, 0.9, f"System monitoring unavailable.\nError: {e}", transform=ax_monitor.transAxes, fontsize=11, va="top", bbox=dict(facecolor="lightcoral"))

    # --- 6. Legend ---
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Idle", markerfacecolor="green", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="To Shelf", markerfacecolor="blue", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="To Station", markerfacecolor="orange", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Returning", markerfacecolor="red", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Low Buffer", markerfacecolor="purple", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Half Full", markerfacecolor="orange", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Nearly Full", markerfacecolor="red", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Disabled", markerfacecolor="dimgray", markersize=10),
    ]
    ax_legend.legend(handles=legend_elements, loc="center", ncol=8, fontsize=12, title_fontproperties={"weight": "bold", "size": 14})

    # --- 7. Final Touches ---
    self.fig.suptitle(f"Warehouse Simulation - Step {step_number}", fontsize=24, weight="bold", y=0.99)
    self.fig.tight_layout(rect=(0, 0, 1, 0.97))

    if self.interactive_mode:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause_time)
    elif frame_dir:
        # Save to file if frame_dir is provided
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        filename = os.path.join(frame_dir, f"frame_{frame:04d}.png")
        self.fig.savefig(filename, dpi=120, bbox_inches="tight")
        if not self.interactive_mode:
            plt.close(self.fig)
            self.fig = None
