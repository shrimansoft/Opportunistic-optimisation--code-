class OrderItem():

    def __init__(self, item_type, creation_time, shelf):
        self.item_type = item_type
        self.creation_time = creation_time
        self.shelf_aloted = shelf
        self.robot_id = None
        self.done_time = None
        self.delay: int = 0

    def shelf_aloted(self, shelf):
        self.shelf_aloted = shelf

    def done(self, time, robot_id):
        self.done_time = time
        self.delay = self.done_time - self.creation_time
        self.robot_id = robot_id
