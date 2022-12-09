# Math
from math import degrees, atan, hypot
import numpy as np
# Data management
from dataclasses import dataclass, field
import json
# tkinter
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
# other
import argparse
import logging
import datetime
from typing import List

VERSION = 0.1
VERSION_TYPE = "development"
AUTHOR = "eiskaffe"  

def get_angle(x: int, y: int) -> float:
    """Returns the angle of a shot given by its x and y coordinates"""
    # Determine the quadrant
    q: int = None
    if x > 0 and y >= 0: 
        q = 1
    elif x <= 0 and y > 0: 
        q = 2
    elif x < 0 and y <= 0: 
        q = 3
    elif x >= 0 and y < 0: 
        q = 4
    
    if x == 0 and y == 0: 
        return 0
    elif x == 0 or y == 0: 
        return (q - 1) * 90
    
    # Calculate the degrees and calculate the correct angle based on the quadrant.
    deg = degrees(abs(atan(y / x)))
    if q == 1: 
        return deg
    else: 
        return (q * 90) - deg

def get_value(hypotenuse: int) -> List[int | None]:
    """Returns the value of a shot given by the shot's hypotenuse and its target's ring_fraction"""
    distance = hypotenuse / RATIO
    # if inner_ten(distance) is False: distance = distance - CALIBER / 2
    for key, item in RING_FRACTION.items():
        if distance <= item: return key
    return 0
 
def date_shot_number_combiner(date: datetime.datetime, shot_number: int) -> str:
    return f"{date.strftime('%y%m%d')}_{shot_number}" 
    
def print_table(matrix: list[list[str]], style="|-+", first_line_head: bool = True, show_line_number: bool = False) -> None:
    """Prints out a table to the stdout from a matrix whose inner lists' length are all equal\n
    STYLE: 0: vertical, 1: horizontal, 2: intersection"""
    # initialise the styles
    vertical, horizontal, intersection = [*style]
    
    if show_line_number:
        L = len(str(len(matrix)))
        print(L)
        for i, row in enumerate(matrix):
            i_len = len(str(i))
            row.insert(0, f"{'0'*(L - i_len)}{i}")
            
    # Get the longest entry in the list by columns
    lengths = np.array([
        [len(string) for string in row]
        for row in matrix
        ])
    # iterate through the columns and save it to a list
    col_lengths = [np.amax(lengths[:,i]) for i in range(lengths.shape[1])]
    
    # Define horizontal_line:
    horizontal_line = intersection
    for max_length in col_lengths:
        horizontal_line += f"{horizontal*(max_length + 2)}{intersection}"
   
    if first_line_head:
        print(horizontal_line)
        head = matrix[0]
        print_row = vertical
        for string, max_length, length in zip(head, col_lengths, lengths[0]):
            print_row += f" {string.upper()}{' '*(max_length - length)} {vertical}"
        print(print_row)
            
        lengths = np.delete(lengths, 0, axis=0)
        matrix.pop(0)
    
    # print the table
    print(horizontal_line)
    for row, row_len in zip(matrix, lengths):
        print_row = vertical
        for string, max_length, length in zip(row, col_lengths, row_len):
            print_row += f" {string}{' '*(max_length - length)} {vertical}"
        print(print_row)
    print(horizontal_line)

@dataclass
class Shot:
    """Defines a shot"""
    x: int
    y: int
    shot_number: int
    date: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(init=False)
    hypotenuse: int = field(init=False)
    angle: float = field(init=False)
    value: float = field(init=False)
    
    def __post_init__(self) -> None:
        self.id = date_shot_number_combiner(self.date, self.shot_number)
        self.hypotenuse = hypot(self.x, self.y)
        self.angle = get_angle(self.x, self.y)
        

    def set_value(self) -> None:
        self.value = get_value(self.hypotenuse)

# TODO
class App(tk.Tk):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        def set_target(x):
            def wrapper():
                print(x)
            return wrapper
        
        self.bind("<Control-q>", lambda *args: self.quit())
                        
        mn = tk.Menu(self) 
        self.config(menu=mn) 
        file_menu = tk.Menu(mn) 
        mn.add_cascade(label='File', menu=file_menu) 
        file_menu.add_command(label='New') 
        # file_menu.add_command(label='Open .csv file', command=open_csv)
        # file_menu.add_command(label='Open images', command=open_img)
        file_menu.add_command(label='Save') 
        file_menu.add_separator() 
        file_menu.add_command(label='About') 
        file_menu.add_separator() 
        file_menu.add_command(label='Exit', command=self.quit) 
        help_menu = tk.Menu(mn) 
        mn.add_cascade(label='Help', menu=help_menu) 
        help_menu.add_command(label='Feedback') 
        help_menu.add_command(label='Contact') 
        
        target_menu = tk.Menu(mn)
        mn.add_cascade(label="Target", menu=target_menu)
        with open("targets.json", "r") as targets:
            targets_json: dict[str: str] = json.loads(targets.read())
        for key in targets_json:
            logger.debug(f"Adding menu target option: {key}")
            target_menu.add_command(label=key.replace("_", " "), command=set_target(key))

def gui() -> None:
    raise NotImplementedError
    app = App()
    app.mainloop()

def cli() -> None:
    
    def target_data(*args) -> None:
        l = len(args)
        if l == 0:
            print(json.dumps(targets_json[target], indent=2))
        elif l == 1 and args[0] == "all":
            print(json.dumps(targets_json, indent=2))
        elif l == 1:
            print(json.dumps(targets_json[args[0]], indent=2))
        elif l > 1:
            logger.exception("target_data only accepts only one positional argument")
        
    def print_help(*args) -> None:
        for command in commands:
            print(f"{command[0]}\t{command[1]}")    
    
    def set_target(*args) -> None:
        if len(args) > 1:
            logger.exception("set_target only accepts only one positional argument")
        elif len(args) == 0:
            logger.exception("set_target requires one positional argument")
        elif args[0] not in targets_json.keys():
            logger.error(f"No target with name {args[0]} was found")
        else:
            nonlocal target
            target = args[0]
        
    print(f"Shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}\n")
    
    target = parsed_args.target
    caliber = parsed_args.caliber
    ratio = parsed_args.ratio
    
    commands = [
        ["target_data", "Prints out the specifications of the selected target. If no target is given, details, the currently selected", target_data],
        ["set_target", f"Sets target. Choices: [{'; '.join(targets_json.keys())}]", set_target],
        ["help", "Prints out this", print_help]
    ]
    
    while True:
        print(f"Target: {'Not set' if target == None else target.replace('_', ' ')};")
        print(f"Caliber: {caliber}mm;\nRatio (between Meyton units and mm): 1mm to {ratio}units")
        print(f"Tip: Type \"help\" for a list of commands")
        
        inp = input("> ").split()
        
        for command in commands:
            if command[0] == inp[0]:
                command[2](*inp[1:])
                break
        else: # NO BREAK
            print(f"Error: command {' '.join(inp)} not found!")
        
        print("\n")
        
        

def main(argv=None) -> None:
    date = datetime.datetime.now().strftime("%Y%m%d")
    
    # TODO: Add GUI mode.

    #! ARGUMENT PARSING
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--caliber", "-c", default=4.5, type=float, help="Sets the caliber in milimeters, defaults to standard 4,5mm diameter of the pellet")
    parent_parser.add_argument("--ratio", default=100, type=int, help="Defines the ratio beetween milimeters and units of distance in the meyton system (Default 1mm = 100 Unit of Distance)")
   
    # Verbosity and logging related arguments
    qvd_parser = parent_parser.add_mutually_exclusive_group()
    qvd_parser.add_argument("--quiet", "-q", action="store_true", help="Only shows errors and exceptions")
    qvd_parser.add_argument("--verbose", "-v", action="store_true", help="Shows the programs actions, warnings, errors and exceptions")
    qvd_parser.add_argument("--debug", action="store_true", help="Shows everything the program does. Only recommended for developers")
    parent_parser.add_argument("--log", default=None, help="Set the logging output, defaults to stdout")
    parent_parser.add_argument("--version", action="version", version=f"shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}", help="Shows the version")

    # main parser
    parser = argparse.ArgumentParser(parents=[parent_parser], prog="Shooter Helper", description="Meyton shot calculator and visualizer from coordinates")
    parser.add_argument("--gui", action="store_true", help="Launches the program in GUI mode")
    parser.add_argument("--csv", help="Reads this file for data")
    with open("targets.json", "r") as targets:
        global targets_json
        targets_json = json.loads(targets.read())
        parser.add_argument("-t", "--target", choices=targets_json.keys(), default=None, help="Sets the target, in which the shots are calculated.")

    global parsed_args
    parsed_args = parser.parse_args() if argv is None else parser.parse_args(argv)
    
    if parsed_args.log is None: logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    else: logging.basicConfig(level=logging.WARNING, filename=parsed_args.log, filemode="w" ,format="%(asctime)s - %(levelname)s - %(message)s")
    global logger
    logger = logging.getLogger(__name__)
    if parsed_args.quiet: logger.setLevel(40)
    elif parsed_args.verbose: logger.setLevel(20)
    elif parsed_args.debug: logger.setLevel(10)   
    
    logger.debug(parsed_args)

    if parsed_args.gui: gui()
    else: cli()

if __name__ == "__main__":
    main()