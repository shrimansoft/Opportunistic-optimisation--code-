class Robot():

    def __init__(self, warehouse, id):
        self.warehouse = warehouse
        self.pickingStation = self.warehouse.picking_stations[0]
        self.robot_id = id
        self.available = True
        # Mode expanation
        # 0: available
        # 1: goint to shelf
        # 2: going to pickup station
        # 3: retruning the shelf to its location.
        self.mode = 0
        self.time_left = 0  # time left in the task.
        self.shelf = None  # which self is above it
        self.current_location = (0, 0)
        self.target_location = self.current_location
        self.shelf_location = None

    def assigne(self, shelf, distance, shelf_location):  # (mode 0 -> 1)
        self.mode = 1
        self.shelf = shelf
        self.shelf_location = shelf_location
        self.target_location = shelf_location
        # print("Assignment id done \t", self.current_location, '\t',
        #       self.target_location)
        self.time_left = distance
        self.available = False
        # print("start>> \t", self.shelf, "distanc:\t", self.time_left)

    def step(self):
        # Move one step towards the target location
        if self.current_location == self.target_location:
            if self.mode == 1:  # (mode 1 -> 2)
                # picking station location.
                self.target_location = self.pickingStation.location
                self.mode = 2
            elif self.mode == 2:  # (mode 2 -> 3)
                self.target_location = self.shelf_location
                order_count = 0
                print(">>> ", self.shelf)
                for i, itemShelfs in enumerate(
                        self.warehouse.itemShelfsBuffer):  # looping over types

                    # looping over shelf assigned to a item type.
                    for itemShelf in itemShelfs:
                        # if the shelf is current shelf
                        if itemShelf == self.shelf:
                            itemShelfs.remove(self.shelf)
                            order_count += 1

                def check_order(order):
                    if order.shelf_aloted == self.shelf:
                        order.done(self.warehouse.time, self.robot_id)
                        self.warehouse.order_compleated.append(order)
                        return False
                    else:
                        return True

                self.warehouse.order_buffer = list(
                    filter(check_order, self.warehouse.order_buffer))
                print("stop>> \t", self.shelf, "order:\t", order_count)
                self.mode = 3

            elif self.mode == 3:  # (mode 3 -> 0)
                self.mode = 0
                self.available = True
                self.shelf = None
                self.shelf_location = None

        if self.time_left > 0:
            self.time_left -= 1

        # Move one step towards the target location
        if self.current_location != self.target_location:
            print("robot id \t", self.robot_id, "roblot current location \t",
                  self.current_location)

            # (We assume that target_location is a tuple like (x, y)
            # and current_location is also a tuple (x, y))
            x, y = self.current_location
            print(self.target_location)
            target_x, target_y = self.target_location

            # Move in the x direction
            if x < target_x:
                x += 1
            elif x > target_x:
                x -= 1
            # Move in the y direction
            elif y < target_y:
                y += 1
            elif y > target_y:
                y -= 1

            self.current_location = (x, y)
