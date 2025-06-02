class PickingStation:
    def __init__(self, warehouse, location, buffer_enabled=True):
        self.warehouse = warehouse
        self.location = location
        self.buffer = []
        self.buffer_enabled = buffer_enabled
        self.default_buffer_size = 8
        self.buffer_size = self.default_buffer_size if buffer_enabled else 0

    def set_buffer_enabled(self, enabled):
        """Enable or disable the buffer dynamically."""
        self.buffer_enabled = enabled
        if enabled:
            self.buffer_size = self.default_buffer_size
        else:
            self.buffer_size = 0
            # Clear the buffer if it's being disabled
            self.buffer.clear()

    def buffer_available(self):
        if not self.buffer_enabled:
            return False
        if len(self.buffer) < self.buffer_size:
            return True
        else:
            return False
