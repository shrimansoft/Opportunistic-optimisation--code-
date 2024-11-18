import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import imageio.v2 as imageio  # For saving the GIF






class Robot():

    def __init__(self,warehouse,id):
        self.warehouse = warehouse
        self.robot_id = id
        self.available = True
        self.mode = 0 # 0: available 1: goint to pick shelf 2: going to pickup station 3: retruning the shelf to its location.
        self.time_left = 0  # time left in the task.
        self.shelf = None  # which self is above it
        self.current_location = (0,0)
        self.target_location = self.current_location
        self.shelf_location = None

    def assigne(self, shelf, distance,shelf_location): # (mode 0 -> 1)
        self.mode = 1
        self.shelf = shelf
        self.shelf_location = shelf_location
        self.target_location = shelf_location
        print("Assignment id done \t",self.current_location,'\t',self.target_location)
        self.time_left = distance
        self.available = False
        print("start>> \t", self.shelf, "distanc:\t", self.time_left)

    def step(self):
        if self.current_location == self.target_location:# Move one step towards the target location
            if self.mode == 1: # (mode 1 -> 2)
                self.target_location = (0,0) # picking station location.
                self.mode = 2
            elif self.mode == 2: # (mode 2 -> 3)
                self.target_location = self.shelf_location

                order_count = 0
                print(">>> ", self.shelf)
                print()
                for i, itemShelfs in enumerate(self.warehouse.itemShelfsBuffer):
                    for itemShelf in itemShelfs:
                        if itemShelf == self.shelf:
                            itemShelfs.remove(self.shelf)
                            self.warehouse.itemBuffer[i] -= 1
                            order_count += 1
                def check_order(order):
                    if order.shelf_aloted == self.shelf:
                        order.done(self.warehouse.time,self.robot_id)
                        self.warehouse.order_compleated.append(order)
                        return False
                    else:
                        return True

                self.warehouse.order_buffer = list(filter(check_order,self.warehouse.order_buffer))
                print("stop>> \t", self.shelf, "order:\t", order_count)
                self.available = True
                self.shelf = None
                self.mode = 3

            elif self.mode == 3: # (mode 3 -> 0)
                self.mode = 0
                self.available = True
                self.shelf = None
                self.shelf_location = None

        if self.time_left > 0:
            self.time_left -= 1

        if self.current_location != self.target_location:# Move one step towards the target location
            print("robot id \t",self.robot_id, "roblot current location \t", self.current_location)

            # (We assume that target_location is a tuple like (x, y) and current_location is also a tuple (x, y))
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


class OrderItem():
    def __init__(self,item_type,creation_time,shelf):
        self.item_type = item_type
        self.creation_time = creation_time
        self.shelf_aloted = shelf
        self.robot_id = None
        self.done_time = None
        self.delay = None

    def shelf_aloted(self,shelf):
        self.shelf_aloted = shelf

    def done(self,time,robot_id):
        self.done_time = time
        self.delay = self.done_time - self.creation_time
        self.robot_id = robot_id

class Warehouse():
    def __init__(self):
        self.time = 0
        self.stock = np.ones(50) * 48 # 50 types of items with 48 of each type.
        self.probabilities = np.random.dirichlet(np.ones(50), size=1)[0] # Assumption from past order distribution.
        self.itemBuffer = np.zeros(50)  # This is the order buffer. 
        self.order_buffer = []
        self.order_compleated = []
        self.itemShelfsBuffer = [[]] * 50
        # self.robots = [Robot(self,1), Robot(self,2)]
        self.robots = [Robot(self,1), Robot(self,2),Robot(self,3)]
        # self.robots = [Robot(self,1)]
        self.distance = np.array([((i % 20) + (i // 20) + 2)
                                  for i in range(400)])

        items = np.repeat(np.arange(0, 50), 48) # fill the wear house
        np.random.shuffle(items)
        shelfs = items.reshape(400, 6)

        self.shelfs = shelfs.tolist()

    def reset(self):
        self.time = 0
        # TODO and many other thing from __init__ when need to be reset. 

    def sample(self):
        """TODO describe function

        :returns: this will return a item from the self.probabili

        """
        return int(
            np.random.choice(np.arange(50), size=1, p=self.probabilities).item())

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
        filteredList = [
            (i, v)
            for i, (v, l) in enumerate(zip(self.distance, availableInShelfs))
            if l
        ]
        shelf, distence = min(filteredList, key=lambda x: x[1])
        return shelf, distence

    def order_step(self):
        self.time += 1
        print("Time ",self.time)
        if (np.random.random() < 0.3):
            available = self.available()
            samples = self.sample()
            if available[samples]:
                self.itemBuffer[samples] += 1
                shelf, distence = self.nearestShelf(samples)
                self.itemShelfsBuffer[samples].append(shelf)
                self.order_buffer.append(OrderItem(samples,self.time,shelf))
                self.shelfs[shelf].remove(samples)
                self.stock[samples] -= 1

    def robot_assigner(self):
        itemShelfsBufferSet = list(set().union(*self.itemShelfsBuffer))

        if len(itemShelfsBufferSet) > 0:
            for robot in self.robots:
                if robot.available:
                    # print(itemShelfsBufferSet)
                    print("Total stock >> ", self.stock.sum())

                    shelf_to_move = int(np.random.choice(itemShelfsBufferSet))

                    robot.assigne(shelf_to_move, 2 * self.distance[shelf_to_move],(shelf_to_move%20 +1,shelf_to_move//20 +1))


    def shelf_plot(self,frame_dir):
        frame = self.time
        shelfs = self.shelfs
        itemShelfsBufferSet = list(set().union(*self.itemShelfsBuffer))
        # Define discrete colormap
        cmap = mcolors.ListedColormap(['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
        norm = mcolors.BoundaryNorm(np.arange(0,8), cmap.N)

        shelf_buffer = np.array([(i in itemShelfsBufferSet) for i in range(400)])
        shelf_buffer_layout = shelf_buffer.reshape(20,20)

        shelf_counts = np.array([len(a) for a in shelfs])
        # shelf_counts = shelfs.sum(axis=1)  # Sum along each shelf's items
        warehouse_layout = shelf_counts.reshape(20, 20)  # Reshape to 20x20 for the warehouse

        # Create the plot for this frame
        fig  = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)

        img1 = ax1.imshow(warehouse_layout, cmap=cmap, norm=norm, interpolation='nearest')
        img2 = ax2.imshow(shelf_buffer_layout,cmap=mcolors.ListedColormap(['#f7fbff','#08519c']), interpolation='nearest')

        def degine(ax,title):
            # Set up the x and y ticks to show 1 to 20
            ax.set_xticks(np.arange(20))
            ax.set_yticks(np.arange(20))
            ax.set_xticklabels(np.arange(1, 21))
            ax.set_yticklabels(np.arange(1, 21))

            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")


            ax.grid(False)

        degine(ax1,"Warehouse Shelf Distribution")
        degine(ax2,"Oder buffer")

        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

        filename = os.path.join(frame_dir, f'frame_{frame:04d}.png')
        plt.savefig(filename)

        plt.close(fig)  # Close the figure to avoid display in notebooks

def main():

    warehouse = Warehouse()
    itemBuffer = warehouse.itemBuffer
    shelfs = warehouse.shelfs
    probabilities = warehouse.probabilities
    # plt.barh(range(50),probabilities)
    # plt.show()

    available = warehouse.available


    # while True:
    for t in range(100):


        warehouse.shelf_plot('shelfs2')

        warehouse.order_step()

        for robot in warehouse.robots:
            robot.step()

        if warehouse.stock.sum() == 0:
            print(t)
            break

        # order exisution

        # order to robot assignment.
        warehouse.robot_assigner()

        t += 1


    # Average delay
    delay = 0
    order_count = 0
    for order in warehouse.order_compleated:
        print("order \t",order.item_type,' Delay: \t',order.delay,"robot: \t",order.robot_id)
        delay += order.delay
        order_count += 1
    print("total order ",order_count," with average delay ",delay/order_count,".")

def expectedTime():
    total_time = 0
    count = 0
    total_prob = 0
    for i in itemShelfsBufferSet:
        dist = distance[i]
        shelf = shelfs[i]

        for item in shelf:
            prob = probabilities[item]
            time = prob * dist  # we are assuming unit speed.
            count += 1
            total_prob += prob
            total_time += time
            # print(dist)
    return total_time / total_prob


if __name__ == "__main__":
    ...
    # main()

main()

