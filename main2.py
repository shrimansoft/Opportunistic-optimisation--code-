import numpy as np
import matplotlib.pyplot as plt


class Robot():

    def __init__(self):
        self.available = True
        self.time_left = 0  # time left in the task.
        self.shelf = None  # which self is above it

    def assigne(self, shelf, distance):
        self.shelf = shelf
        self.time_left = distance
        self.available = False
        print("start>> \t", self.shelf, "distanc:\t", self.time_left)

    def step(self, itemShelfsBuffer, itemBuffer):
        if self.time_left == 0 and (not self.available):
            order_count = 0
            print(">>> ", self.shelf)
            print()
            for i, itemShelfs in enumerate(itemShelfsBuffer):
                for itemShelf in itemShelfs:
                    if itemShelf == self.shelf:
                        itemShelfs.remove(self.shelf)
                        itemBuffer[i] -= 1
                        order_count += 1
            print("stop>> \t", self.shelf, "order:\t", order_count)
            self.available = True
            self.shelf = None
        if self.time_left > 0:
            self.time_left -= 1


class Wearhouse():

    def __init__(self):
        self.stock = np.ones(50) * 48
        self.probabilities = np.random.dirichlet(np.ones(50), size=1)[0]
        self.itemBuffer = np.zeros(50)  # don't use it.
        self.itemShelfsBuffer = [[]] * 50
        self.robots = [Robot(), Robot()]
        self.distance = np.array([((i % 20) + (i // 20) + 2)
                                  for i in range(400)])

        # fill the wear house
        items = np.repeat(np.arange(0, 50), 48)
        np.random.shuffle(items)
        shelfs = items.reshape(400, 6)
        self.shelfs = shelfs.tolist()
        ...

    def sample(self):
        return int(
            np.random.choice(np.arange(50), size=1, p=self.probabilities))

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
        if (np.random.random() < 0.3):
            available = self.available()
            samples = self.sample()
            if available[samples]:
                self.itemBuffer[samples] += 1
                shelf, distence = self.nearestShelf(samples)

                self.itemShelfsBuffer[samples].append(shelf)
                self.shelfs[shelf].remove(samples)

                self.stock[samples] -= 1


def main():

    wearhouse = Wearhouse()
    stock = wearhouse.stock
    itemBuffer = wearhouse.itemBuffer
    #shelfs (but not using this)
    itemShelfsBuffer = wearhouse.itemShelfsBuffer
    shelfs = wearhouse.shelfs
    #filling the wearhouse with random itesm (order history)
    probabilities = wearhouse.probabilities
    # plt.barh(range(50),probabilities)
    # plt.show()

    available = wearhouse.available

    distance = wearhouse.distance

    robot_is_available = wearhouse.robots[0].available
    robot_time_left_in_the_task = wearhouse.robots[0].time_left
    robot_shelf = wearhouse.robots[0].shelf
    t = 0

    while True:
        # for t in range(100):

        wearhouse.order_step()

        wearhouse.robots[0].step(wearhouse.itemShelfsBuffer,
                                 wearhouse.itemBuffer)

        itemShelfsBufferSet = list(set().union(*itemShelfsBuffer))
        if len(itemShelfsBufferSet) == 0 and stock.sum() == 0:
            print(t)
            break

        # order exisution
        itemShelfsBufferSet = list(set().union(*itemShelfsBuffer))

        if len(itemShelfsBufferSet) > 0 and wearhouse.robots[0].available:
            # print(itemShelfsBufferSet)
            print("Sthock>> ", stock.sum())

            shelf_to_move = int(np.random.choice(itemShelfsBufferSet))

            wearhouse.robots[0].assigne(shelf_to_move,
                                        2 * distance[shelf_to_move])
        t += 1


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
    main()
