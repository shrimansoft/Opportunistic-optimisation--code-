import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .ploter import ploter

# Set backend before importing pyplot to avoid TclError issues
matplotlib.use("TkAgg")  # Use TkAgg backend for better interactive support


from typing import List

from .order import OrderItem
from .picking_station import PickingStation
from .robot import Robot


class Warehouse:
    def __init__(self, seed=None, buffer_enabled=True):
        if seed is not None:
            np.random.seed(seed)

        self.time = 0
        self.buffer_enabled = buffer_enabled
        self.probabilities = np.random.dirichlet(np.ones(50), size=1)[
            0
        ]  # Assumption from past order distribution.
        self.distance = np.array([((i % 20) + (i // 20) + 2) for i in range(400)])

        self.stock = np.ones(50) * 48  # 50 types of items with 48 of each type.
        items = np.repeat(np.arange(0, 50), 48)  # fill the wear house
        np.random.shuffle(items)
        shelfs = items.reshape(400, 6)
        self.shelfs = shelfs.tolist()

        self.order_buffer: List[OrderItem] = []
        self.order_compleated: List[OrderItem] = []
        self.itemShelfsBufferSet = set()

        self.picking_stations = [
            PickingStation(self, (0, 14), buffer_enabled=self.buffer_enabled),
            PickingStation(self, (0, 10), buffer_enabled=self.buffer_enabled),
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

        # Initialize figure for interactive plotting
        self.fig = None
        self.interactive_mode = False

    def buffer_update(self, shelf, picking_station: PickingStation):
        """
        Redistributes items between shelf and buffer based on demand probabilities.
        Most demanded items go to buffer, rest to shelf. Only works if buffers are enabled.
        """

        # Skip buffer operations if buffers are disabled
        if not self.buffer_enabled or not picking_station.buffer_enabled:
            return

        # Combine and sort items by demand probability (descending)
        all_items = self.shelfs[shelf] + picking_station.buffer
        if not all_items:
            return

        sorted_items = sorted(
            all_items, key=lambda item: self.probabilities[item], reverse=True
        )

        # Clear and redistribute based on max buffer capacity
        self.shelfs[shelf].clear()
        picking_station.buffer.clear()

        max_capacity = picking_station.buffer_size
        picking_station.buffer.extend(sorted_items[:max_capacity])
        self.shelfs[shelf].extend(sorted_items[max_capacity : max_capacity * 2])

    def reset(self):
        """Reset the warehouse to initial state for a new simulation."""
        self.time = 0

        # Reset stock to initial state
        self.stock = np.ones(50) * 48  # 50 types of items with 48 of each type

        # Reinitialize shelf layout
        items = np.repeat(np.arange(0, 50), 48)  # fill the warehouse
        np.random.shuffle(items)
        shelfs = items.reshape(400, 6)
        self.shelfs = shelfs.tolist()

        # Clear order queues
        self.order_buffer.clear()
        self.order_compleated.clear()
        self.itemShelfsBufferSet.clear()

        # Reset robots to initial state
        for robot in self.robots:
            robot.available = True
            robot.mode = 0
            robot.time_left = 0
            robot.shelf = None
            robot.current_location = (0, 0)
            robot.target_location = (0, 0)
            robot.shelf_location = None

        # Reset picking station buffers
        for station in self.picking_stations:
            station.buffer.clear()

    def sample(self):
        """TODO describe function

        :returns: this will return a item from the self.probabili

        """
        return int(np.random.choice(np.arange(50), size=1, p=self.probabilities).item())

    def available(self):
        # Check both shelf stock and picking station buffers
        availability = list(map(bool, self.stock))

        # Only check buffers if they are enabled
        if self.buffer_enabled:
            # Also check if items are available in any picking station buffer
            for i in range(len(availability)):
                if not availability[
                    i
                ]:  # Only check buffers if item not available on shelves
                    # Check all picking stations for this item
                    for picking_station in self.picking_stations:
                        if i in picking_station.buffer:
                            availability[i] = True
                            break

        return availability

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
            (i, v) for i, (v, l) in enumerate(zip(distance, availableInShelfs)) if l
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
                # Check if the item is available in any picking station buffer (only if buffers are enabled)
                item_found_in_buffer = False
                if self.buffer_enabled:
                    for station in self.picking_stations:
                        if samples in station.buffer:
                            # Item found in buffer - create order and fulfill immediately
                            order = OrderItem(
                                samples, self.time, None
                            )  # No shelf needed
                            order.done(
                                self.time, None
                            )  # Completed immediately, no robot needed
                            self.order_compleated.append(order)
                            station.buffer.remove(samples)
                            self.stock[samples] -= 1
                            item_found_in_buffer = True
                            print(
                                f"Order for item {samples} fulfilled immediately from picking station buffer"
                            )
                            print("Total stock >> ", self.stock.sum())
                            break

                # If item not found in buffer, create regular order
                if not item_found_in_buffer:
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

    def enable_interactive_plot(self):
        """Enable interactive plotting mode."""
        self.interactive_mode = True
        plt.ion()

    def disable_interactive_plot(self):
        """Disable interactive plotting mode."""
        self.interactive_mode = False
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def average_delay(self):
        delay = 0
        order_count = 0
        for order in self.order_compleated:
            # print(
            #     "order \t",
            #     order.item_type,
            #     " Delay: \t",
            #     order.delay,
            #     "robot: \t",
            #     order.robot_id,
            # )
            delay += order.delay
            order_count += 1
        for order in self.order_buffer:
            delay += self.time - order.creation_time
            order_count += 1
        if order_count == 0:
            return 0
        else:
            return delay / order_count

    def enhanced_plot(self, frame_dir=None, step_number=None, pause_time=0.1):
        return ploter(self, frame_dir=None, step_number=None, pause_time=0.1)

    def set_buffer_enabled(self, enabled):
        """Enable or disable all picking station buffers."""
        self.buffer_enabled = enabled
        for station in self.picking_stations:
            station.set_buffer_enabled(enabled)

    def set_station_buffer_enabled(self, station_index, enabled):
        """Enable or disable a specific picking station buffer."""
        if 0 <= station_index < len(self.picking_stations):
            self.picking_stations[station_index].set_buffer_enabled(enabled)
        else:
            raise IndexError(f"Station index {station_index} out of range")
