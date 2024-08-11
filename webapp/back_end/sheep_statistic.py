from distance_calculation import DistanceCalculation

class SheepStatistic:
    def __init__(self) -> None:
        self.__sheep_coordinate = {}
        self.__sheep_move_by_time = []
        self.__sheep_distance_moved = {}
        self.__myDistance_calculator = DistanceCalculation()
        
    def update_coordinate(self, sheep_id, new_coordinate):
        if sheep_id not in self.__sheep_coordinate:
            self.__sheep_coordinate[sheep_id] = {
                'prev': [0, 0, 0, 0],
                'curr': new_coordinate,
                'distance_move': 0,
            }
        else:
            distance_move = self.__myDistance_calculator.cal_distance(tuple(new_coordinate), tuple(self.__sheep_coordinate[sheep_id]['curr']))
            self.__sheep_coordinate[sheep_id] = {
                'prev': self.__sheep_coordinate[sheep_id]['curr'],
                'curr': new_coordinate,
                'distance_move': self.__sheep_coordinate[sheep_id]['distance_move'] + distance_move
            }
            
    def update_statistic(self):
        new_statistic = {}
        for k, value in self.__sheep_coordinate.items():
            new_statistic[k] = value['distance_move']
        self.__sheep_move_by_time.append(new_statistic)

    def get_coordinate(self):
        return self.__sheep_coordinate
    
    def get_moved_distance(self):
        return self.__sheep_move_by_time
    
    def get_current_moved_distance(self):
        new_statistic = {}
        for k, value in self.__sheep_coordinate.items():
            new_statistic[k] = value['distance_move']
        return new_statistic
    
    def update_statistic(self, id, distance):
        if id not in self.__sheep_distance_moved:
            self.__sheep_distance_moved[id] = 0
        else:
            self.__sheep_distance_moved[id] += distance
            
    def get_statistic(self):
        return self.__sheep_distance_moved

    def get_current_statistic(self):
        return self.__sheep_distance_moved
    def reset(self):
        self.__sheep_coordinate = {}
        self.__sheep_move_by_time = []
        self.__sheep_distance_moved = {}