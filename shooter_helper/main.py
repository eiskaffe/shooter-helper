# Math
from math import degrees, atan, hypot, dist, floor
from statistics import median, mean
import numpy as np
import cv2
# Data management
from dataclasses import dataclass, field, InitVar
import json
# tkinter
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
# other
import argparse
import logging
import datetime
from typing import List, Iterable, Callable
import os

VERSION = 0.1
VERSION_TYPE = "development"
AUTHOR = "eiskaffe"  

@dataclass    
class Target:
    name: str
    data: dict
    values: dict[float, float] = field(init=False)
    int_values: dict[int, int] = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        self.values = {float(f"10.{10 - i - 1}"): round(((self.data["ten"] / 10 * (i + 1)) / 2), 6) for i in range(10)}
        self.values.update({(100 - i - 1) / 10: round(((self.data["ten"] + self.data["quotient"] / 10 * (i + 1)) / 2), 6) for i in range(100)})
        self.int_values = {int(key): value for key, value in self.values.items() if key.is_integer()}
        logger.debug(f"{self.name} target initialized with {self.values=}")

@dataclass
class Shot:
    """Defines a shot"""
    id: str
    date: datetime.datetime
    x: int
    y: int
    _ratio: InitVar[int] = 1
    
    def __post_init__(self, _ratio: int) -> None:
        self.x = int(self.x * _ratio)
        self.y = int(self.y * _ratio)
        self.date = datetime.datetime(*map(int, self.date.split(".")))
    
    @property
    def shot_no(self) -> int:
        return int(self.id.split("_")[-1])
    
    @property
    def hypotenuse(self) -> float:
        """The distance of the shot from the center"""
        return hypot(self.x, self.y)
    
    @property
    def angle(self) -> float:
        """Returns the angle of a shot given by its x and y coordinates"""
        # Determine the quadrant
        q: int = None
        if self.x > 0 and self.y >= 0: q = 1
        elif self.x <= 0 and self.y > 0: q = 2
        elif self.x < 0 and self.y <= 0: q = 3
        elif self.x >= 0 and self.y < 0: q = 4
        
        if self.x == 0 and self.y == 0: return 0
        elif self.x == 0 or self.y == 0: return (q - 1) * 90

        # Calculate the degrees and calculate the correct angle based on the quadrant.
        deg = degrees(abs(atan(self.y / self.x)))
        if q == 1: return deg
        return (q * 90) - deg

    def get_value(self, target: Target, integer_only: bool = False) -> float:
        """Returns the corresponding value of a shot based on the target"""
        # TODO inner ten weird behavior
        if integer_only:
            for value, distance in target.values.items():
                if self.hypotenuse <= distance: return value
        else:
            for value, distance in target.int_values.items():
                if self.hypotenuse <= distance: return value
        return 0

def load_shots_from_csv(filename: str, ratio: int = 1, sep: str = ";") -> list[Shot]:
    with open(filename, "r") as csvfile:
        return [Shot(*line.strip().split(sep))
                for line in csvfile]
            
def draw(shots: list[Shot], target: Target, base_image: cv2.Mat = None, ratio: int = 1) -> cv2.Mat:
    """Draws the shots to a file\n
    N is the image size in pixels: 2^N (Default: 2^12-1 = 2048)"""
    
    img = np.zeros((target.data["card_size"][0]*RATIO, target.data["card_size"][1]*RATIO, 3), np.uint8)
    img[::] = (192, 21, 0)
    xmid 
    
    
    font_size = target.data["font_size"]
    img = np.zeros((target.data["card_size"][0]*RATIO, target.data["card_size"][1]*RATIO, 3), np.uint8)
    img[::] = (192, 21, 0)
    center = (target.data["card_size"][0] // 2 * RATIO, target.data["card_size"][1] // 2 * RATIO)
    xmid, ymid = center
    quotient = int(target.data["quotient"]*RATIO) // 2
    ring_thickness = int(target.data["ring_thickness"]*RATIO)
    img = cv2.circle(img, center, 0, (255, 255, 255), int(RINGS[-1]*2*RATIO))
    img = cv2.circle(img, center, 0, (90, 144, 21), int(RING_FRACTION[target.data["black"]]*2*RATIO))
    if logger.level == 10:
        img = cv2.circle(img, center, 0, (0, 0, 255), 20)
        img = cv2.line(img, (xmid, 0), (xmid, target.data["card_size"][1]*RATIO), (0, 0, 255), 5)
        img = cv2.line(img, (0, ymid), (target.data["card_size"][0]*RATIO, ymid), (0, 0, 255), 5)
    for i, radius in enumerate(RINGS):
        if 10 - i < target.data["black"]: color = (0, 0, 0)
        else: color = (235, 235, 235)
        img = cv2.circle(img, center, int(radius*RATIO), color, ring_thickness)
        
        # TODO
        # if 10 - i <= target.data["numbers"][0] and 10 - i >= target.data["numbers"][1]:
        #     (w, h), (we, he) = get_real_text_size(str(10-i), cv2.FONT_HERSHEY_PLAIN, font_size, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - (w + we)//2, ymid - i*quotient + h + h//2 - target.data["inner_ten"]*RATIO), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - (w + we)//2, ymid + i*quotient - h//2 + target.data["inner_ten"]*RATIO), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - i*quotient + w - target.data["inner_ten"]*RATIO, ymid + (h + he)//2), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid + i*quotient - we - w + target.data["inner_ten"], ymid + (h + he)//2), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        
    for i, shot in enumerate(shots):
        xmid = shot.x + target.data["card_size"][0]//2*RATIO if shot.x >= 0 else target.data["card_size"][0]//2*RATIO - abs(shot.x)
        ymid = target.data["card_size"][0]//2*RATIO - shot.y if shot.y >= 0 else abs(shot.y) + target.data["card_size"][0]//2*RATIO
        img = cv2.circle(img, (xmid, ymid), 0, (0, 0, 0), int(CALIBER*RATIO))
        img = cv2.circle(img, (xmid, ymid), int(CALIBER*RATIO)//2, (255, 255, 255), int(target.data["ring_thickness"]*RATIO))
        (label_width, label_height), baseline = cv2.getTextSize(str(i + 1), cv2.FONT_HERSHEY_PLAIN, font_size//1.4, int(font_size*1.2))
        img = cv2.putText(img, str(i + 1), (xmid - label_width//2, ymid + label_height//2), cv2.FONT_HERSHEY_PLAIN, font_size//1.4, (0, 255, 255), int(font_size*1.2))
    
    if analyze:
        xmid, ymid = center
        (x_mn, y_mn, mn), (x_med, y_med, med) = mean_median_of_shots(shots)
        print(mean_median_of_shots(shots))
        img = cv2.circle(img, (x_mn + xmid, y_mn + ymid), mn, (255, 119, 51), 50)
        img = cv2.circle(img, (x_med + xmid, y_med + ymid), med, (255, 0, 212), 50)
    
    img = cv2.resize(img, (2 ** N, 2 ** N))
    if write:
        cv2.imwrite(f"{datetime.datetime.now().strftime('%Y%m%d')}.jpg", img) 
    if show:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 1024, 1024)
        cv2.imshow("img", img)
        cv2.waitKey(0)
    
    return img             
        
def main(argv = None) -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    global logger
    logger = logging.getLogger(__name__)
    
    targets = {}
    with open("targets.json", "r", encoding="utf-8") as targets_file:
        targets_json: dict[str, Target] = json.loads(targets_file.read())
        for key, data in targets_json.items():
            targets[key] = Target(key, data)
            logger.debug(f"{key} target successfully loaded")
            print(targets[key].int_values)
    
    shots = load_shots_from_csv("test_dataset.csv")
    print(shots[1])

# def get_value(hypotenuse: int, ratio, ring_fraction) -> List[int | None]:
#     """Returns the value of a shot given by the shot's hypotenuse and its target's ring_fraction"""
#     distance = hypotenuse / ratio
#     # if inner_ten(distance) is False: distance = distance - CALIBER / 2
#     #TODO Implement binary search
#     for key, item in ring_fraction.items():
#         if distance <= item: return key
#     return 0
 
# def date_shot_number_combiner(date: datetime.datetime, shot_number: int) -> str:
#     return f"{date.strftime('%y%m%d')}_{shot_number}" 
    
# def print_table(matrix: list[list[str]], style="|-+", first_line_head: bool = True, show_line_number: bool = False) -> None:
#     """Prints out a table to the stdout from a matrix whose inner lists' length are all equal\n
#     STYLE: 0: vertical, 1: horizontal, 2: intersection"""
#     # initialise the styles
#     vertical, horizontal, intersection = [*style]
    
#     if show_line_number:
#         L = len(str(len(matrix)))
#         print(L)
#         for i, row in enumerate(matrix):
#             i_len = len(str(i))
#             row.insert(0, f"{'0'*(L - i_len)}{i}")
            
#     # Get the longest entry in the list by columns
#     lengths = np.array([
#         [len(string) for string in row]
#         for row in matrix
#         ])
#     # iterate through the columns and save it to a list
#     col_lengths = [np.amax(lengths[:,i]) for i in range(lengths.shape[1])]
    
#     # Define horizontal_line:
#     horizontal_line = intersection
#     for max_length in col_lengths:
#         horizontal_line += f"{horizontal*(max_length + 2)}{intersection}"
   
#     if first_line_head:
#         print(horizontal_line)
#         head = matrix[0]
#         print_row = vertical
#         for string, max_length, length in zip(head, col_lengths, lengths[0]):
#             print_row += f" {string.upper()}{' '*(max_length - length)} {vertical}"
#         print(print_row)
            
#         lengths = np.delete(lengths, 0, axis=0)
#         matrix.pop(0)
    
#     # print the table
#     print(horizontal_line)
#     for row, row_len in zip(matrix, lengths):
#         print_row = vertical
#         for string, max_length, length in zip(row, col_lengths, row_len):
#             print_row += f" {string}{' '*(max_length - length)} {vertical}"
#         print(print_row)
#     print(horizontal_line)

# def get_distances_of_shots(shots: list[Shot]) -> list[float]:
#     """Given a N item long list of shots, returns a N-1 item long list
#     of floats of the distances between the shots."""
    
#     x_list: list[int] = [shot.x for shot in shots]
#     y_list: list[int] = [shot.y for shot in shots]
    
#     return [
#         dist((x_list[i], y_list[i]), (x_list[i+1], y_list[i+1]))
#         for i in range(len(shots)-1)
#         ]

# def get_consistency(shots: list[Shot]) -> tuple[list[float], list[float]]:
#     """Returns relative and absolute constistency"""
    
#     distances = get_distances_of_shots(shots)
#     maximal = max(distances)

#     # Linear Distribution
#     return [distance / maximal for distance in distances]  

# def key_sum(A: Iterable, key: Callable) -> float:
#     """A sum function with a key, like in max or min."""
#     summa = 0.0
#     for a in A:
#         summa += key(a)
#     return summa
        

# # TODO
# class App(tk.Tk):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
        
#         def set_target(x):
#             def wrapper():
#                 print(x)
#             return wrapper
        
#         self.bind("<Control-q>", lambda *args: self.quit())
                        
#         mn = tk.Menu(self) 
#         self.config(menu=mn) 
#         file_menu = tk.Menu(mn) 
#         mn.add_cascade(label='File', menu=file_menu) 
#         file_menu.add_command(label='New') 
#         # file_menu.add_command(label='Open .csv file', command=open_csv)
#         # file_menu.add_command(label='Open images', command=open_img)
#         file_menu.add_command(label='Save') 
#         file_menu.add_separator() 
#         file_menu.add_command(label='About') 
#         file_menu.add_separator() 
#         file_menu.add_command(label='Exit', command=self.quit) 
#         help_menu = tk.Menu(mn) 
#         mn.add_cascade(label='Help', menu=help_menu) 
#         help_menu.add_command(label='Feedback') 
#         help_menu.add_command(label='Contact') 
        
#         target_menu = tk.Menu(mn)
#         mn.add_cascade(label="Target", menu=target_menu)
#         with open("targets.json", "r") as targets:
#             targets_json: dict[str: str] = json.loads(targets.read())
#         for key in targets_json:
#             logger.debug(f"Adding menu target option: {key}")
#             target_menu.add_command(label=key.replace("_", " "), command=set_target(key))

# def calculate_target(value) -> tuple[dict, dict]:
#     """Returns the ring fraction and the target data, both as dictionaries"""
#     with open("targets.json", "r") as targets:
#         logger.debug(f"Selected value: {value}")
#         targets_json: dict[str: str] = json.loads(targets.read())
#         target.data = targets_json[value]
#         logger.debug(f"{target.data=}")
    
#     ring_fraction: dict[float: float] = {float(f"10.{10 - i - 1}"): round(((target.data["ten"] / 10 * (i + 1)) / 2), 6) for i in range(10)}
#     ring_fraction.update({(100 - i - 1) / 10: round(((target.data["ten"] + target.data["quotient"] / 10 * (i + 1)) / 2), 6) for i in range(100)})
#     logger.debug(f"{ring_fraction=}")
    
#     return ring_fraction, target.data      
        
# def gui() -> None:
#     raise NotImplementedError
#     app = App()
#     app.mainloop()

# def cli() -> None:
    
#     def get_target.data(*args) -> None:
#         l = len(args)
#         if l == 0:
#             print(json.dumps(targets_json[target], indent=2))
#         elif l == 1 and args[0] == "all":
#             print(json.dumps(targets_json, indent=2))
#         elif l == 1:
#             print(json.dumps(targets_json[args[0]], indent=2))
#         elif l > 1:
#             logger.exception("target.data only accepts only one positional argument")
        
#     def print_help(*args) -> None:
#         for command in commands:
#             print(f"{command[0]}\t{command[1]}")    
    
#     def set_target(*args) -> None:
#         if len(args) > 1:
#             logger.exception("set_target only accepts only one positional argument")
#         elif len(args) == 0:
#             logger.exception("set_target requires one positional argument")
#         elif args[0] not in targets_json.keys():
#             logger.error(f"No target with name {args[0]} was found")
#         else:
#             nonlocal target, ring_fraction, target.data
#             target = args[0]
#             ring_fraction, target.data = calculate_target(target)
    
#     def list_targets(*args) -> None:
#         for key in targets_json:
#             print(key)
        
#     def read_csv(*args) -> None:
#         if target == None:
#             logger.exception("No target is set!")
#         elif len(args) > 1:
#             logger.exception("set_target only accepts only one positional argument")
#         elif len(args) == 0:
#             logger.exception("set_target requires one positional argument")
#         elif not os.path.exists(args[0]):
#             logger.exception(f"File {args[0]!r} does not exist.")    
        
#         else:
#             with open(args[0], "r") as csv:
#                 nonlocal days
#                 days = []
#                 current_shots: list[Shot] = []
#                 current_day: datetime.datetime = None
#                 for shot in csv:
#                     data = shot.strip().split(";")
#                     date = datetime.datetime(*map(int, data[1].split(".")))
#                     s: Shot = None
#                     if len(data) == 4:
#                         s = Shot(int(data[2]), int(data[3]), data[0], date)
#                         s.set_value(ratio, ring_fraction)
#                     else:
#                         s = Shot(int(data[2]), int(data[3]), data[0], date, *map(float, data[4:])) 
#                     logger.debug(s)
                    
#                     if current_day == None:
#                         current_shots.append(s)
#                         current_day = date
#                     elif current_day == date:
#                         current_shots.append(s)
#                     else:
#                         days.append(Day(current_shots, target.data, current_day))
#                         current_day = date
#                         current_shots = [s]
                        
#                 days.append(Day(current_shots, target.data, current_day)) # Append last day
                
            
          
        
#     print(f"Shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}\n")
    
#     target: str = parsed_args.target
#     caliber: int = parsed_args.caliber
#     ratio: int = parsed_args.ratio
#     csv_file: str = None
    
#     ring_fraction, target.data = (None, None) if target == None else calculate_target(target)
#     days: list[Day] = []
    
#     commands = [
#         ["target.data", "Prints out the specifications of the selected target. If no target is given, details, the currently selected", get_target.data],
#         ["set_target", f"Sets target. Choices: [{'; '.join(targets_json.keys())}]", set_target],
#         ["list_targets", "Lists all available targets you can choose from", list_targets],
#         ["read", "Reads the selected csv file and lets you use the data from it.", read_csv],
#         ["help", "Prints out this", print_help],
#         ["exit", "Exits the program", lambda *args: exit()]
        
#     ]
    
#     while True:
#         print(f"Target: {'Not set' if target == None else target.replace('_', ' ')};")
#         print(f"Caliber: {caliber}mm;\nRatio (between Meyton units and mm): 1mm to {ratio}units")
#         print(f"Tip: Type \"help\" for a list of commands")
        
#         inp = input("> ").split()
        
#         for command in commands:
#             if command[0] == inp[0]:
#                 command[2](*inp[1:])
#                 break
#         else: # NO BREAK
#             print(f"Error: command {inp[0]} not found!")
        
#         print("\n")
        
        

# def main(argv=None) -> None:
#     date = datetime.datetime.now().strftime("%Y%m%d")
    
#     # TODO: Add GUI mode.

#     #! ARGUMENT PARSING
#     parent_parser = argparse.ArgumentParser(add_help=False)
#     parent_parser.add_argument("--caliber", "-c", default=4.5, type=float, help="Sets the caliber in milimeters, defaults to standard 4,5mm diameter of the pellet")
#     parent_parser.add_argument("--ratio", default=100, type=int, help="Defines the ratio beetween milimeters and units of distance in the meyton system (Default 1mm = 100 Unit of Distance)")
   
#     # Verbosity and logging related arguments
#     qvd_parser = parent_parser.add_mutually_exclusive_group()
#     qvd_parser.add_argument("--quiet", "-q", action="store_true", help="Only shows errors and exceptions")
#     qvd_parser.add_argument("--verbose", "-v", action="store_true", help="Shows the programs actions, warnings, errors and exceptions")
#     qvd_parser.add_argument("--debug", action="store_true", help="Shows everything the program does. Only recommended for developers")
#     parent_parser.add_argument("--log", default=None, help="Set the logging output, defaults to stdout")
#     parent_parser.add_argument("--version", action="version", version=f"shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}", help="Shows the version")

#     # main parser
#     parser = argparse.ArgumentParser(parents=[parent_parser], prog="Shooter Helper", description="Meyton shot calculator and visualizer from coordinates")
#     parser.add_argument("--gui", action="store_true", help="Launches the program in GUI mode")
#     parser.add_argument("--csv", help="Reads this file for data")
#     with open("targets.json", "r") as targets:
#         global targets_json
#         targets_json = json.loads(targets.read())
#         parser.add_argument("-t", "--target", choices=targets_json.keys(), default=None, help="Sets the target, in which the shots are calculated.")

#     global parsed_args
#     parsed_args = parser.parse_args() if argv is None else parser.parse_args(argv)
    
#     if parsed_args.log is None: logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
#     else: logging.basicConfig(level=logging.WARNING, filename=parsed_args.log, filemode="w" ,format="%(asctime)s - %(levelname)s - %(message)s")
#     global logger
#     logger = logging.getLogger(__name__)
#     if parsed_args.quiet: logger.setLevel(40)
#     elif parsed_args.verbose: logger.setLevel(20)
#     elif parsed_args.debug: logger.setLevel(10)   
    
#     logger.debug(parsed_args)

#     if parsed_args.gui: gui()
#     else: cli()

if __name__ == "__main__":
    main()