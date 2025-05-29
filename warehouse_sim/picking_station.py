class PickingStation:
    def __init__(self, warehouse, location):
        self.warehouse = warehouse
        self.location = location
        self.buffer = []
        self.buffer_size = 8

    def buffer_available(self):
        if len(self.buffer) < self.buffer_size:
            return True
        else:
            return False
