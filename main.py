import numpy as np
from typing import List,Union,Tuple,Optional

class Item():
    def __init__(self,itemId,itemType,itemProb):
        self.itemId = ...
        self.itemType = ...
        self.itemProb = ...
        self.shelf = ... # shelf reference
        self.bin = ... # wher in the shelf? 

class ItemType():
    def __init__(self):
        self.typeId = ...
        self.size= ...
        self.typeProb = ...
        self.available = ...
        self.quantity = ...
        self.items = ... # this will be an array of items. 



class Shelfs():
    def __init__(self,id: int,location: Optional[Tuple[int,int]]) -> None:
        self.id : int =id
        self.location: Optional[Tuple[int,int]] = location
        self.items: Tuple[int,int,int,int,int,int] = ... # they will store the item id

class Locations():
    def __init__(self):
        self.locations = np.zeros((20,20))
        self.shelfs = [Shelfs(i,None) for i in range(200)]


class Robots():
    def __init__(self):
        self.noOfRobots = ...
        self.robots = ...


class Wearhouse():

    def __init__(self,layout,robotes, items):
        """TODO describe function

        :returns: 

        """
        self.locations = Locations();
        self.items = Items();
        self.robots = Robots();
        self.pikingStations = ...
        self.orderBuffer = ...

    def reset(self):
        ...

    def step(self,action):
        ...

    def ExpectedTime(self,orderHistory):
        ...





def main():
    wearhouse = Wearhouse()

    





if __name__ == "__main__":
    main()
