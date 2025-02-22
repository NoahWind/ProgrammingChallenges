class Prisoner:
    def __init__(self, number, x_position, y_position, x_next_position, y_next_position, tries_left, next_box, current_box):
        self.number = number
        self.x_position = x_position
        self.y_position = y_position
        self.x_next_position = x_next_position
        self.y_next_position = y_next_position
        self.tries_left = tries_left
        self.next_box = next_box
        self.current_box = current_box

class Box:
    def __init__(self, number):
        self.number = number
        self.next = None