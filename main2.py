

import numpy as np
import matplotlib.pyplot as plt


# %% 
itemTypes = np.arange(0,50)
print(itemTypes)
print(itemTypes.dtype)

stock = np.ones(50)*48
print(stock)
print(stock.dtype)

itemBuffer = np.zeros(50)
print(itemBuffer)

#shelfs (but not using this)
itemShelfsBuffer = [[]]*50

#shelfs (but not using this)
shelfs = np.empty(400,dtype=object)
shelfs[:] = [[]]
print(shelfs)




# fill the wear house
items = np.repeat(np.arange(0,50),48)
print(items)

np.random.shuffle(items)
print(items)

shelfs = items.reshape(400,6)
shelfs = shelfs.tolist()
shelfs



#filling the wearhouse with random itesm (order history)
probabilities = np.random.dirichlet(np.ones(50), size=1)[0]

plt.barh(range(50),probabilities)
plt.show()
print(probabilities)


# sample the oreder randomly. 

samples = np.random.choice(np.arange(50), size=10, p=probabilities)
print(samples)


# orde only for stock


print(stock)

stock[1:30] = 0

available = list(map(bool,stock))

print(probabilities*available)



# shelf distance
distance = np.array([((i%20)+(i//20)+2) for i in range(400)])
print(distance)

# shelf buffer



def itemInShelfs(n):
    return list(map(lambda x:sum([1 for i in x if i == n]), shelfs))


def nearestShelf(n):
    availableInShelfs= list(map(bool,itemInShelfs(n)))
    # print(availableInShelfs)

    filteredList = [(i,v) for i,(v,l) in enumerate(zip(distance,availableInShelfs)) if l]
    # print(filteredList)

    shelf,distence = min(filteredList,key=lambda x: x[1])
    # print(shelf)
    return shelf,distence

print(nearestShelf(9))

print(shelfs)
# %%
# while True:

robot_is_available = True
robot_time_left_in_the_task = 0
robot_shelf = None
t=0

# for t in range(100):
while True:
    if(np.random.random()<0.3):
        available = list(map(f,stock))
        samples = int(np.random.choice(np.arange(50), size=1, p=probabilities ))
        if available[samples]:
            itemBuffer[samples] += 1
            shelf,distence = nearestShelf(samples)
            # print(type(samples))
            itemShelfsBuffer[samples].append(shelf)
            shelfs[shelf].remove(samples)
            # print(itemShelfsBuffer[samples][0])
            stock[samples] -= 1
            # print(itemBuffer)

    if robot_time_left_in_the_task == 0 and (not robot_is_available):
        order_count = 0
        for i,itemShelfs in enumerate(itemShelfsBuffer):
            for itemShelf in itemShelfs:
                if itemShelf == robot_shelf:
                    itemShelfs.remove(robot_shelf)
                    itemBuffer[i] -= 1
                    order_count += 1
        print("shelf>> \t",robot_shelf,"stoped at \t", t,"order:\t",order_count)
        robot_is_available =True
        robot_shelf = None
    if len(itemShelfsBufferSet) == 0 and stock.sum() == 0:
        print(i)
        break

    # order exisution
    itemShelfsBufferSet =  list(set().union(*itemShelfsBuffer))
    # select a shelf

    if robot_time_left_in_the_task > 0:
        robot_time_left_in_the_task -= 1

    if len(itemShelfsBufferSet) > 0 and robot_is_available:
        shelf_to_move = int(np.random.choice(itemShelfsBufferSet))
        # print(itemShelfsBufferSet)
        # print(shelf_to_move)
        robot_shelf = shelf_to_move
        robot_is_available = False
        robot_time_left_in_the_task = 2*distance[shelf_to_move]
        print("shelf>> \t",robot_shelf," started at \t", t,"distanc:\t",robot_time_left_in_the_task,"total stock\t", stock.sum())

    t+=1
    # you have one robot

# %%

# itemShelfsBufferSet =  list(set().union(*itemShelfsBuffer))




# print(itemShelfsBuffer)
# print(itemShelfsBufferSet)
# print([shelfs[i] for i in itemShelfsBufferSet])
# print([distance[i] for i in itemShelfsBufferSet])


# calculate the expected time

def expectedTime():
    total_time = 0
    count = 0
    total_prob = 0
    for i in itemShelfsBufferSet:
        dist = distance[i]
        shelf = shelfs[i]

        for item in shelf:
            prob = probabilities[item]
            time = prob * dist # we are assuming unit speed.
            count += 1
            total_prob += prob
            total_time += time
            # print(dist)
    return total_time/total_prob


























