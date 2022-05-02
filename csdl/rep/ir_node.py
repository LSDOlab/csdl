class IRNode:

    def __init__(self):
        self.abs_name: str = ''
        self.times_visited = 0
        self.name: str = ''

    def incr_times_visited(self):
        self.times_visited += 0
