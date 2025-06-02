import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from typing import List, Dict

from .picking_station import PickingStation
from .order import OrderItem
from .robot import Robot


class Warehouse:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.time = 0
        self.probabilities = np.random.dirichlet(
            np.ones(50), size=1)[0]  # Assumption from past order distribution.
        self.distance = np.array([((i % 20) + (i // 20) + 2)
                                  for i in range(400)])

        self.stock = np.ones(
            50) * 48  # 50 types of items with 48 of each type.
        items = np.repeat(np.arange(0, 50), 48)  # fill the wear house
        np.random.shuffle(items)
        shelfs = items.reshape(400, 6)
        self.shelfs = shelfs.tolist()

        self.order_buffer: List[OrderItem] = []
        self.order_compleated: List[OrderItem] = []
        self.itemShelfsBufferSet = set()

        self.picking_stations = [
            PickingStation(self, (0, 14)),
            PickingStation(self, (0, 10))
        ]
        self.robots = [
            Robot(self, 1),
            Robot(self, 2),
            Robot(self, 3),
            Robot(self, 4),
            Robot(self, 5),
            Robot(self, 6),
            # Robot(self,7), Robot(self,8),Robot(self,9),
        ]
        # self.robots = [Robot(self,1)]



    def buffer_update(self, shelf, picking_station: PickingStation):
        """
        Redistributes items between shelf and buffer based on demand probabilities.
        Most demanded items go to buffer, rest to shelf  """
        # Combine and sort items by demand probability (descending)
        all_items = self.shelfs[shelf] + picking_station.buffer
        if not all_items:
            return
        
        sorted_items = sorted(all_items, key=lambda item: self.probabilities[item], reverse=True)
        
        # Clear and redistribute based on max buffer capacity
        self.shelfs[shelf].clear()
        picking_station.buffer.clear()
        
        max_capacity = picking_station.buffer_size
        picking_station.buffer.extend(sorted_items[:max_capacity])
        self.shelfs[shelf].extend(sorted_items[max_capacity:max_capacity*2])


    def reset(self):
        self.time = 0
        # TODO and many other thing from __init__ when need to be reset.

    def sample(self):
        """TODO describe function

        :returns: this will return a item from the self.probabili

        """
        return int(
            np.random.choice(np.arange(50), size=1,
                             p=self.probabilities).item())

    def available(self):
        return list(map(bool, self.stock))

    def itemInShelfs(self, n):
        """TODO describe function

        :param n: item number
        :returns: a list showing which shelf have how much item n.

        """
        return list(map(lambda x: sum([1 for i in x if i == n]), self.shelfs))

    def nearestShelf(self, n):
        availableInShelfs = list(map(bool, self.itemInShelfs(n)))
        distance = [
            0 if i in self.itemShelfsBufferSet else self.distance[i]
            for i in range(len(self.distance))
        ]
        filteredList = [
            (i, v) for i, (v, l) in enumerate(zip(distance, availableInShelfs))
            if l
        ]
        shelf, distance = min(filteredList, key=lambda x: x[1])
        return shelf, distance

    def order_step(self):
        self.time += 1
        if self.time % 1000 == 0:
            print("Total average_dealy: ", self.average_delay())
        if np.random.random() < 0.3:
            available = self.available()
            samples = self.sample()
            if available[samples]:
                shelf, distence = self.nearestShelf(samples)
                self.itemShelfsBufferSet.add(shelf)
                self.order_buffer.append(OrderItem(samples, self.time, shelf))
                self.shelfs[shelf].remove(samples)
                print("Total stock >> ", self.stock.sum())
                self.stock[samples] -= 1

    def robot_assigner(self):
        itemShelfsBufferSet = self.itemShelfsBufferSet

        if len(itemShelfsBufferSet) > 0:
            for robot in self.robots:
                if robot.available:
                    # print(itemShelfsBufferSet)
                    if len(self.itemShelfsBufferSet) > 0:
                        shelf_to_move = self.itemShelfsBufferSet.pop()
                        # self.itemShelfsBufferSet.remove(shelf_to_move)
                        robot.assigne(
                            shelf_to_move,
                            2 * self.distance[shelf_to_move],
                            (shelf_to_move % 20 + 1, shelf_to_move // 20 + 1),
                        )

    def shelf_plot(self, frame_dir):
        frame = self.time
        shelfs = self.shelfs
        itemShelfsBufferSet = self.itemShelfsBufferSet
        # Define discrete colormap
        cmap = mcolors.ListedColormap([
            "#f7fbff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#3182bd",
            "#08519c",
        ])
        norm = mcolors.BoundaryNorm(np.arange(0, 8), cmap.N)

        shelf_counts = np.array([len(a) for a in shelfs])
        # shelf_counts = shelfs.sum(axis=1)  # Sum along each shelf's items
        warehouse_layout = shelf_counts.reshape(
            20, 20)  # Reshape to 20x20 for the warehouse

        # Create the plot for this frame
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        img1 = ax1.imshow(warehouse_layout,
                          cmap=cmap,
                          norm=norm,
                          interpolation="nearest")

        # Plot picking stations
        for station_idx, station in enumerate(self.picking_stations):
            station_y, station_x = station.location
            station_x -= 1
            station_y -= 1

            # Plot picking station as a large square marker
            ax1.plot(station_y,
                     station_x,
                     "s",
                     markersize=12,
                     color="purple",
                     markeredgecolor="black",
                     markeredgewidth=2)
            ax2.plot(station_y,
                     station_x,
                     "s",
                     markersize=12,
                     color="purple",
                     markeredgecolor="black",
                     markeredgewidth=2)

            # Add label for picking station with station number
            ax1.text(station_y,
                     station_x,
                     f"{station_idx}",
                     color="white",
                     fontsize=8,
                     ha="center",
                     va="center",
                     weight="bold")
            ax2.text(station_y,
                     station_x,
                     f"{station_idx}",
                     color="white",
                     fontsize=8,
                     ha="center",
                     va="center",
                     weight="bold")

        # Plot robot locations

        for robot in self.robots:
            robot_y, robot_x = robot.current_location
            # itemShelfsBufferSet = itemShelfsBufferSet.union({robot.shelf})

            robot_x -= 1
            robot_y -= 1

            # Define color based on robot mode
            if robot.mode == 0:
                robot_color = "green"  # Idle mode (available)
            elif robot.mode == 1:
                robot_color = "blue"  # Going to pick shelf
            elif robot.mode == 2:
                robot_color = "orange"  # Going to pickup station
            elif robot.mode == 3:
                robot_color = "red"  # Returning the shelf

            if robot.shelf_location is not None:
                shelf_y, shelf_x = robot.shelf_location
                ax2.text(
                    shelf_y - 1,
                    shelf_x - 1,
                    f"{robot.robot_id}",
                    color="white",
                    fontsize=5,
                    ha="center",
                    va="center",
                )
                ax2.plot(shelf_y - 1,
                         shelf_x - 1,
                         "D",
                         markersize=8,
                         color="#08519c")  # Circle marker for robot

            # Plot robot's current location with the appropriate color
            ax1.plot(robot_y, robot_x, "o", markersize=8,
                     color=robot_color)  # Circle marker for robot
            ax2.plot(robot_y, robot_x, "o", markersize=8,
                     color=robot_color)  # Circle marker for robot

            # Display robot ID and mode at the robot's position
            ax1.text(
                robot_y,
                robot_x,
                f"{robot.robot_id}",
                color="white",
                fontsize=5,
                ha="center",
                va="center",
            )
            ax2.text(
                robot_y,
                robot_x,
                f"{robot.robot_id}",
                color="white",
                fontsize=5,
                ha="center",
                va="center",
            )

        shelf_buffer = np.array([(i in itemShelfsBufferSet)
                                 for i in range(400)])
        shelf_buffer_layout = shelf_buffer.reshape(20, 20)

        img2 = ax2.imshow(
            shelf_buffer_layout,
            cmap=mcolors.ListedColormap(["#f7fbff", "#08519c"]),
            interpolation="nearest",
        )

        def degine(ax, title):
            # Set up the x and y ticks to show 1 to 20
            ax.set_xticks(np.arange(20))
            ax.set_yticks(np.arange(20))
            ax.set_xticklabels(np.arange(1, 21))
            ax.set_yticklabels(np.arange(1, 21))
            ax.set_xlim(-1.5, 20.5)
            ax.set_ylim(-1.5, 20.5)

            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.grid(False)

        degine(ax1, "Warehouse Shelf Distribution")
        degine(ax2, "Oder buffer")

        # Display additional information (Total Stock, Orders in Progress, etc.)
        total_stock = self.stock.sum()
        total_orders = len(self.order_buffer)
        completed_orders = len(self.order_compleated)

        # Shelf details for each robot
        robot_shelf_info = []
        for robot in self.robots:
            if robot.shelf:
                robot_shelf_info.append(
                    f"R{robot.robot_id} carrying Shelf {robot.shelf}")
            else:
                robot_shelf_info.append(f"R{robot.robot_id} idle")

        # Format text information
        robot_shelf_text = "\n".join(robot_shelf_info)

        # Place the details at the bottom of the plot
        ax1.text(
            0.5,
            -0.1,
            f"Total Stock: {total_stock} | Orders in Progress: {total_orders} | Completed Orders: {completed_orders}",
            ha="center",
            va="top",
            transform=ax1.transAxes,
            fontsize=12,
            color="black",
            weight="bold",
        )

        ax1.text(
            0.5,
            -0.2,
            robot_shelf_text,
            ha="center",
            va="top",
            transform=ax1.transAxes,
            fontsize=10,
            color="black",
        )

        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

        filename = os.path.join(frame_dir, f"frame_{frame:04d}.png")
        plt.savefig(filename)

        plt.close(fig)  # Close the figure to avoid display in notebooks

    def average_delay(self):
        delay = 0
        order_count = 0
        for order in self.order_compleated:
            # print("order \t",order.item_type,' Delay: \t',order.delay,"robot: \t",order.robot_id)
            delay += order.delay
            order_count += 1
        for order in self.order_buffer:
            delay += self.time - order.creation_time
            order_count += 1
        if order_count == 0:
            return 0
        else:
            return delay / order_count
