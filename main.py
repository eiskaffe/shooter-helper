# Math
from math import sqrt, atan, degrees, floor, dist, hypot
from statistics import median, mean
import numpy as np
# Image processing
import imglibrary
import cv2
import pytesseract
from pytesseract import Output
# Data management
import json
import csv
# GUI
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
# Other
import logging
import os
from alive_progress import alive_bar
from typing import List
from dataclasses import dataclass, field
import argparse
import datetime

# METADATA
VERSION = 0.1
VERSION_TYPE = "development"
AUTHOR = "eiskaffe"  

def hypot(x: int, y: int) -> float:
    """Returns the hypotenuse of a shot given by its x and y coordinates"""
    return sqrt(x ** 2 + y ** 2)

def get_angle(x: int, y: int) -> float:
    """Returns the angle of a shot given by its x and y coordinates"""
    q: int = None
    if x > 0 and y >= 0: q = 1
    elif x <= 0 and y > 0: q = 2
    elif x < 0 and y <= 0: q = 3
    elif x >= 0 and y < 0: q = 4
    
    if x == 0 and y == 0: return 0
    elif x == 0 or y == 0: return (q - 1) * 90
    else:
        deg = degrees(abs(atan(y / x)))
        if q == 1: return deg
        else: return (q * 90) - deg

def get_value(hypotenuse: int) -> List[int | None]:
    """Returns the value of a shot given by the shot's hypotenuse and its target's ring_fraction"""
    distance = hypotenuse / RATIO
    if inner_ten(distance) is False: distance = distance - CALIBER / 2
    for key, item in RING_FRACTION.items():
        if distance <= item: return key
    return 0

def inner_ten(hypotenuse: int) -> bool:
    """Returns a true if the shot is a inner ten. If not returns False"""
    if hypotenuse + MEASURING_EDGE / 2 <= RINGS[MEASURING_RING]: return True
    else: return False

def longest_string_in_matrix(matrix: list[list]) -> list[int]:
    """Returns the longest element in a 2D list or matrix in each column"""
    columns = len(matrix[0])
    column_length = [0 for _ in range(columns)]
    for row in matrix:
        for i, value in enumerate(row):
            if len(str(value)) > column_length[i]: column_length[i] = len(str(value))
    return column_length

def printTable(matrix:list[list[str]], head:list[str]=[], style="|-+") -> None:
    """Prints out a table to the stdout from a matrix whose inner lists' length are all equal\n
    STYLE: 0: vertical, 1: horizontal, 2: intersection"""
    head = [head]
    column_length = longest_string_in_matrix(matrix + head)
    vertical, horizontal, intersection = [*style]
    for value in column_length:
        intersection = f"{intersection}{horizontal*(value + 2)}{style[2]}"
    print(intersection)
        
    if head != [[]]:
        print_row = vertical
        for column, max_length in zip(head[0], column_length):
            print_row = f"{print_row} {column.upper()}{' '*(max_length - len(column) + 1)}{vertical}"
        print(print_row)
        print(intersection)
        
    for row in matrix:
        print_row = vertical
        for column, max_length in zip(row, column_length):
            print_row = f"{print_row} {column}{' '*(max_length - len(column) + 1)}{vertical}"
        print(print_row)
    print(intersection)

# TODO  
def manual_ten_insertion():
    raise NotImplementedError()

# TODO
def image_parser(img_path: str, allowed_characters: str = "-0123456789", config: str = r"--psm 12"):
    raise NotImplementedError("Image Parsing is not implemented yet")
    factor = 13
    if os.path.exists(img_path) is False: raise FileNotFoundError(f"Given image file ({img_path}) does not exist.")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (2**factor, 2**factor), interpolation=cv2.INTER_CUBIC)
    
    img = imglibrary.noiseRemoval(img, iterations=30)
    img = imglibrary.deskew(img)
    img = imglibrary.grayscale(img, a=16, b=500)
    img = imglibrary.thickFont(img, iterations=32)
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1024, 1024)
    cv2.imshow("img", img)
    cv2.waitKey(0) 

    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)

    confidence = [conf for conf in data['conf'] if conf != -1]
    logger.debug(f"CONFIDENCE {confidence}")
    text = [text for text in data['text'] if text != ""]
    logger.debug(f"TEXT {text}")

    # if logger.level == 10:
    #     amount_boxes = len(data["text"])
    #     for i in range(amount_boxes):
    #         if float(data["conf"][i]) > 50:
    #             (x, y, width, height) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
    #             img = cv2.rectangle(img, (x, y), (x+width, y+height), (0,255,0), 2)
    #     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow('img', 600, 800)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     lis = data["text"]
    #     lis[:] = [item for item in lis if item != '']
    #     print(lis)

    # Filtering:
    allowed_characters = [*allowed_characters]
    filtered_list = [text for conf, text in zip(confidence, text, strict=True) if conf != -1]
    filtered_list = [word for word in filtered_list if all(letter in allowed_characters for letter in word)]
    logger.debug(f"FILTERED LIST {filtered_list}")
    
    # final_list = filtered_list.copy()
    # while len([word for word in final_list if all(letter in allowed_characters for letter in word)]) != 20:
    #     printTable([[str(i), str(val), f"{conf}% {' !' if conf < 80 else ''}"] for i, (val, conf) in enumerate(zip(filtered_list, confidence))], head=["id", "value", "confidence"])
    #     input("Please select an ID that needs to be edited.")
    
    # for conf, text in zip(confidence, data['text']):
    #     if conf != -1: continue
    
    # # print(allowed_characters)
    # dictionary = data['text']
    # final_list = [word for word in dictionary if all(letter in allowed_characters for letter in word)]
    # logger.debug(f"IMAGE FILTERED LIST ({img_path}): {final_list}")
    
    # if len(final_list) != 20:
    #     logger.error(f"Image ({img_path}) filtered list length is not 20 ({len(final_list)})! Please edit the image to only the coordinates be visible!")
    #     if not args.disable_errors:
    #         print("Please review the following data, due to image to data issues")
    #         printTable([[i, ] for i, val in enumerate(final_list)], head=["ID", "X", "Y"])
    #     else:
    #         raise ValueError("Got more or less values than 20 in the list for the image parsing") 

    return [[final_list[i], final_list[i+1]] for i in range(0, len(final_list), 2)]

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
        self.value = get_value(self.hypotenuse)

# TODO
@dataclass
class Day:
    """Defines a day. A day contains N number of Shots"""
    shots: list[Shot]
    target: str
    date: datetime.datetime = field(default_factory=datetime.datetime.now)
    summa: tuple[int] = field(init=False)
    average: tuple[int] = field(init=False)
    abs_consistency: list[int] = field(init=False)
    rel_consistency: list[int] = field(init=False)
    mean_of_shots: tuple[int] = field(init=False)
    median_of_shots: tuple[int] = field(init=False)
    shot_distances: list[int] = field(init=False)
    number_of_shots: int = field(init=False)
    
    def draw(self, a = 0, b = -1):
        if b == -1: b = len(self.shots)
    
    def __post_init__(self):
        self.number_of_shots = len(self.shots)
        
        print(self.shots)
        
        a, b = 0, 0
        for shot in self.shots:
            a += floor(shot.value)
            b += shot.value
        self.summa = (a, b)
        self.average = (a / self.number_of_shots, b / self.number_of_shots)
        
        self.shot_distances = get_distances_of_shots(self.shots)
        maximal = hypot(TARGET_DATA["card_size"][0]*RATIO, TARGET_DATA["card_size"][1]*RATIO)
        self.abs_consistency = [distance / maximal for distance in self.shot_distances]
        self.rel_consistency = get_consistency(self.shots)
    
def get_real_text_size(
    string: str, font: int, font_size: int, 
    thickness: int, line_type:int=cv2.LINE_8, write_proof=False
    ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Returns the real size of a text and the errors
    Return: (width, height), (width_error, height_error)"""
    img_w = len(string) * font_size * 20
    img_h = font_size * 50
    img = np.zeros((img_h, img_w), np.uint8)
    img = cv2.putText(img, string, (100, img_h - 100), font, font_size, (255, 255, 255), thickness, line_type)
    indexes = np.where(img==255)
    height = np.max(indexes[0]) - np.min(indexes[0])
    width = np.max(indexes[1]) - np.min(indexes[1])
    
    width_error = np.min(indexes[1]) - 100
    height_error = np.max(indexes[0]) - (img_h - 100)
    
    if write_proof:
        img = np.zeros((img_h, img_w), np.uint8)
        img = cv2.rectangle(img, (100, img_h - 100), (100 + width, (img_h - 100) - height), (127, 127, 127), 1, cv2.LINE_8)
        img = cv2.rectangle(img, (np.min(indexes[1]), np.max(indexes[0])), (np.max(indexes[1]), np.min(indexes[0])), (200, 200, 200), 1, cv2.LINE_8) 
        img = cv2.putText(img, string, (100, img_h - 100), font, font_size, (255, 255, 255), thickness, line_type)
        cv2.imwrite("width_and_height_proof.jpg", img)
    
    return (width, height), (width_error, height_error)

def mean_median_of_shots(shots: list[Shot]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return: tuple[x-mean, y-mean, hyp-mean], tuple[x-median, y-median, hyp-median]"""
    x_list: list[int] = [shot.x for shot in shots]
    y_list: list[int] = [shot.y for shot in shots]
    
    x_mn = int(mean(x_list))
    y_mn = int(mean(y_list))
    hyp_mn = int(mean([dist((x, y), (x_mn, y_mn)) for x, y in zip(x_list, y_list)]))
    
    x_med = int(median(x_list))
    y_med = int(median(y_list))
    hyp_med = int(median([dist((x, y), (x_med, y_med)) for x, y in zip(x_list, y_list)]))
    
    return (x_mn, y_mn, hyp_mn), (x_med, y_med, hyp_med)

def get_distances_of_shots(shots: list[Shot]) -> list[float]:
    """Given a N item long list of shots, returns a N-1 item long list
    of floats of the distances between the shots."""
    
    x_list: list[int] = [shot.x for shot in shots]
    y_list: list[int] = [shot.y for shot in shots]
    
    return [
        dist((x_list[i], y_list[i]), (x_list[i+1], y_list[i+1]))
        for i in range(len(shots)-1)
        ]

def get_consistency(shots: list[Shot]) -> tuple[list[float], list[float]]:
    """Returns relative and absolute constistency"""
    
    distances = get_distances_of_shots(shots)
    maximal = max(distances)

    # Linear Distribution
    return [distance / maximal for distance in distances]

def draw(shots: list[Shot], font_size: int = None, N: int = 12, write: bool = False, show: bool = False, analyze: bool = False) -> cv2.Mat:
    """Draws the shots to a file\n
    N is the image size in pixels: 2^N (Default: 2^12 = 2048)"""
    
    font_size = TARGET_DATA["font_size"]
    img = np.zeros((TARGET_DATA["card_size"][0]*RATIO, TARGET_DATA["card_size"][1]*RATIO, 3), np.uint8)
    img[::] = (192, 21, 0)
    center = (TARGET_DATA["card_size"][0] // 2 * RATIO, TARGET_DATA["card_size"][1] // 2 * RATIO)
    xmid, ymid = center
    quotient = int(TARGET_DATA["quotient"]*RATIO) // 2
    ring_thickness = int(TARGET_DATA["ring_thickness"]*RATIO)
    img = cv2.circle(img, center, 0, (255, 255, 255), int(RINGS[-1]*2*RATIO))
    img = cv2.circle(img, center, 0, (90, 144, 21), int(RING_FRACTION[TARGET_DATA["black"]]*2*RATIO))
    if logger.level == 10:
        img = cv2.circle(img, center, 0, (0, 0, 255), 20)
        img = cv2.line(img, (xmid, 0), (xmid, TARGET_DATA["card_size"][1]*RATIO), (0, 0, 255), 5)
        img = cv2.line(img, (0, ymid), (TARGET_DATA["card_size"][0]*RATIO, ymid), (0, 0, 255), 5)
    for i, radius in enumerate(RINGS):
        if 10 - i < TARGET_DATA["black"]: color = (0, 0, 0)
        else: color = (235, 235, 235)
        img = cv2.circle(img, center, int(radius*RATIO), color, ring_thickness)
        
        # TODO
        # if 10 - i <= TARGET_DATA["numbers"][0] and 10 - i >= TARGET_DATA["numbers"][1]:
        #     (w, h), (we, he) = get_real_text_size(str(10-i), cv2.FONT_HERSHEY_PLAIN, font_size, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - (w + we)//2, ymid - i*quotient + h + h//2 - TARGET_DATA["inner_ten"]*RATIO), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - (w + we)//2, ymid + i*quotient - h//2 + TARGET_DATA["inner_ten"]*RATIO), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid - i*quotient + w - TARGET_DATA["inner_ten"]*RATIO, ymid + (h + he)//2), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        #     img = cv2.putText(img, str(10-i), (xmid + i*quotient - we - w + TARGET_DATA["inner_ten"], ymid + (h + he)//2), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        
    for i, shot in enumerate(shots):
        xmid = shot.x + TARGET_DATA["card_size"][0]//2*RATIO if shot.x >= 0 else TARGET_DATA["card_size"][0]//2*RATIO - abs(shot.x)
        ymid = TARGET_DATA["card_size"][0]//2*RATIO - shot.y if shot.y >= 0 else abs(shot.y) + TARGET_DATA["card_size"][0]//2*RATIO
        img = cv2.circle(img, (xmid, ymid), 0, (0, 0, 0), int(CALIBER*RATIO))
        img = cv2.circle(img, (xmid, ymid), int(CALIBER*RATIO)//2, (255, 255, 255), int(TARGET_DATA["ring_thickness"]*RATIO))
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
    
def evaluateDay(lst):
    ...
    
def date_shot_number_combiner(date: datetime.datetime, shot_number: int) -> str:
    return f"{date.strftime('%y%m%d')}_{shot_number}"

def set_target(value):
    with open("targets.json", "r") as targets:
        logger.debug(f"Selected value: {value}")
        targets_json: dict[str: str] = json.loads(targets.read())
        global TARGET_DATA
        TARGET_DATA = targets_json[value]
        logger.debug(f"TARGET DATA: {TARGET_DATA}")
        del targets_json
        
        global RINGS
        RINGS = [round(((TARGET_DATA["ten"] + TARGET_DATA["quotient"] * i) / 2), 6) for i in range(10)]
        logger.debug(f"RINGS: {RINGS}")

        global RING_FRACTION
        RING_FRACTION = {float(f"10.{10 - i - 1}"): round(((TARGET_DATA["ten"] / 10 * (i + 1)) / 2), 6) for i in range(10)}
        RING_FRACTION.update({(100 - i - 1) / 10: round(((TARGET_DATA["ten"] + TARGET_DATA["quotient"] / 10 * (i + 1)) / 2), 6) for i in range(100)})
        logger.debug(f"RING FRACTION: {RING_FRACTION}")
        
        global MEASURING_EDGE, MEASURING_RING
        MEASURING_EDGE = TARGET_DATA["inner_ten_measuring_edge"]
        MEASURING_RING = TARGET_DATA["inner_ten_measuring_ring"]
        
        global TARGET_SET
        TARGET_SET = value

# TODO
class App(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.days: list[Day] = list()
        
        self.geometry("1000x500")
        self.title("Shooter Helper")
        self.iconbitmap(os.path.join(os.path.dirname(__file__), "assets", "icon.ico"))
        
        def open_csv():
            if TARGET_SET:
                csv_file = filedialog.askopenfilename()
                with open(csv_file, "r", encoding="utf-8") as inf:
                    csvreader = csv.reader(inf, delimiter=CSV_DELIMETER)
                    # id    date    x   y   (value   hypotenuse  angle)
                    # 0     1       2   3   (4       5           6)
                    shots: list[Shot] = [
                        Shot(
                            x=int(shot[2]),
                            y=int(shot[3]),
                            shot_number=int(shot[0].split("_")[1]),
                            date=datetime.datetime(
                                year=int(shot[1].split(".")[0]),
                                month=int(shot[1].split(".")[1]),
                                day=int(shot[1].split(".")[2])
                            )
                        )
                        for shot in csvreader
                    ]
                print(shots)
                current_date = shots[0].date
                print(current_date, "\n----------")
                sublist = []
                for shot in shots:
                    if shot.date == current_date:
                        sublist.append(shot)
                    else:
                        self.days.append(Day(sublist, TARGET_SET, current_date))
                        current_date = shot.date
                        sublist = []
                self.days.append(Day(sublist, TARGET_SET, current_date))              
                logger.debug(self.days)
                
            else:
                logger.error("Target not set")
    
        def open_img():
            raise NotImplementedError("Opening images is not implemented")
        
        def set_target_(value):
            with open("targets.json", "r") as targets:
                logger.debug(f"Selected value: {value}")
                targets_json: dict[str: str] = json.loads(targets.read())
                global TARGET_DATA
                TARGET_DATA = targets_json[value]
                logger.debug(f"TARGET DATA: {TARGET_DATA}")
                del targets_json
                
                global RINGS
                RINGS = [round(((TARGET_DATA["ten"] + TARGET_DATA["quotient"] * i) / 2), 6) for i in range(10)]
                logger.debug(f"RINGS: {RINGS}")

                global RING_FRACTION
                RING_FRACTION = {float(f"10.{10 - i - 1}"): round(((TARGET_DATA["ten"] / 10 * (i + 1)) / 2), 6) for i in range(10)}
                RING_FRACTION.update({(100 - i - 1) / 10: round(((TARGET_DATA["ten"] + TARGET_DATA["quotient"] / 10 * (i + 1)) / 2), 6) for i in range(100)})
                logger.debug(f"RING FRACTION: {RING_FRACTION}")
                
                global MEASURING_EDGE, MEASURING_RING
                MEASURING_EDGE = TARGET_DATA["inner_ten_measuring_edge"]
                MEASURING_RING = TARGET_DATA["inner_ten_measuring_ring"]
                
                global TARGET_SET
                TARGET_SET = value
                
        self.bind("<Control-q>", lambda *args: self.quit())
                        
        mn = tk.Menu(self) 
        self.config(menu=mn) 
        file_menu = tk.Menu(mn) 
        mn.add_cascade(label='File', menu=file_menu) 
        file_menu.add_command(label='New') 
        file_menu.add_command(label='Open .csv file', command=open_csv)
        file_menu.add_command(label='Open images', command=open_img)
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
            # WHY??????????????????????????????????????????????????????????????????????????????????????????????
            target_menu.add_command(label=key.replace("_", " "), command=lambda: set_target(key))

        self.my_img = ImageTk.PhotoImage(Image.open("20221021.jpg").resize((500, 500)))
        self.my_label = tk.Label(image=self.my_img)
        self.my_label.pack()
            
    def print_contents(self, event):
        print("Hi. The current entry content is:", self.contents.get())     
    
# SUB-COMMANDS
# TODO
def image_sub_command():
    image_data = [image_parser(img_path) for img_path in args.imagefiles]
    date = args.date if isinstance(args.date, datetime.datetime) else datetime.datetime(
                    year=int(args.date.split(".")[0]),
                    month=int(args.date.split(".")[1]),
                    day=int(args.date.split(".")[2])
                )
    shots = [Shot(x=int(shot[2]), 
                y=int(shot[3]), 
                shot_number=int(shot[0].split("_")[1]), 
                date=date)
            for shot in image_data
            ]
    
    if args.debug:
        printTable([list(map(str, tens)) for tens in lst], head=["id", "date", "x", "y", "value", "angle", "hyp"], firstlinehead=False)
        
    if args.target is not None: draw(lst, target_data)
    
    with open(args.output, "a") as outfile:
        for shot in [list(map(str, tens)) for tens in lst]:
            outfile.write(CSV_DELIMETER.join(shot) + "\n")
    
# TODO  
def day_sub_command():
    with open(args.csvfile, "r") as infile:
        csvreader = csv.reader(infile, delimiter=CSV_DELIMETER)
        date = args.date if isinstance(args.date, datetime.datetime) else datetime.datetime(
                year=int(args.date.split(".")[0]),
                month=int(args.date.split(".")[1]),
                day=int(args.date.split(".")[2])
            )
        # id    date    x   y   value   hypotenuse  angle
        # 0     1       2   3   4       (5)         (6)
        shots = [Shot(x=int(shot[2]), 
                    y=int(shot[3]), 
                    shot_number=int(shot[0].split("_")[1]), 
                    date=date)
                    for shot in csvreader
                    if shot[1] == args.date
                    ]
    draw(shots, write=True, analyze=True)
    
    maximal = hypot(TARGET_DATA["card_size"][0]*RATIO, TARGET_DATA["card_size"][1]*RATIO)
    abs_consistency = [distance / maximal for distance in get_distances_of_shots(shots)]
    rel_consistency = get_consistency(shots)
    
    app = App()
    app.mainloop()  
                
def gui_sub_command():
    app = App()
    app.mainloop()
    
def main(argv=None):
    # Other
    date = datetime.datetime.now().strftime("%Y%m%d")
    
    #! ARGUMENT PARSING
    parent_parser = argparse.ArgumentParser(add_help=False)
    # Important arguments
    # with open("targets.json", "r") as targets:
    #     targets_json = json.loads(targets.read())
    #     parent_parser.add_argument("-t", "--target", required=True, choices=targets_json.keys(), help="Sets the target, in which the shots are calculated (Required)")
    parent_parser.add_argument("--caliber", "-c", default=4.5, type=int, help="Sets the caliber in milimeters, defaults to standard 4,5mm diameter of the pellet")
    parent_parser.add_argument("--ratio", default=100, type=int, help="Defines the ratio beetween milimeters and units of distance in the meyton system (Default 1mm = 100 Unit of Distance)")
    # Verbosity and logging related arguments
    qvd_parser = parent_parser.add_mutually_exclusive_group()
    qvd_parser.add_argument("--quiet", "-q", action="store_true", help="Only shows errors and exceptions")
    qvd_parser.add_argument("--verbose", "-v", action="store_true", help="Shows the programs actions, warnings, errors and exceptions")
    qvd_parser.add_argument("--debug", action="store_true", help="Shows everything the program does. Only recommended for developers")
    parent_parser.add_argument("--log", default=None, help="Set the logging output, defaults to stdout")
    parent_parser.add_argument("--version", action="version", version=f"shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}", help="Shows the version")
    # Other
    parser = argparse.ArgumentParser(parents=[parent_parser], prog="Shooter Helper", description="Meyton shot calculator and visualizer from coordinates")

    # Sub-commands
    subparsers = parser.add_subparsers(help="Sub-commands")
    
    img_parser = subparsers.add_parser("image", help="Image input related", parents=[parent_parser])
    img_parser.add_argument("imagefiles", nargs="*", help="Reads the shots' coordinates from the images. The images order are set by their names. ")
    img_parser.add_argument("--output", "-o", default=f"{date}-shots.csv", help=f"Sets the output file. File format delimited to csv. Defaults to {date}-shots.csv")
    with open("targets.json", "r") as targets:
            targets_json = json.loads(targets.read())
            img_parser.add_argument("-t", "--target", choices=targets_json.keys(), required=True, help="Sets the target, in which the shots are calculated. (Required)")
    img_parser.add_argument("--disable-errors", action="store_true", help="When this argument is given, the program terminates, and will not try to get user input to solve the issue")
    img_parser.add_argument("--date, -d", default=datetime.datetime.now().strftime('%Y.%m.%d'), help=f"Set the date to parse, defaults to today's date as ({datetime.datetime.now().strftime('%Y.%m.%d')})")
    img_parser.set_defaults(func=image_sub_command, gui=False)
    
    csv_parser = subparsers.add_parser("calculate", help="lorem ipsum", parents=[parent_parser])
    
    day_parser = subparsers.add_parser("day", help="lorem ipsum", parents=[parent_parser])
    day_parser.add_argument("csvfile", help="Reads this file for data")
    day_parser.add_argument("date", nargs="?", default=datetime.datetime.now().strftime('%Y.%m.%d'), help=f"Set the date to parse, defaults to today's date as ({datetime.datetime.now().strftime('%Y.%m.%d')})")
    with open("targets.json", "r") as targets:
        targets_json = json.loads(targets.read())
        day_parser.add_argument("-t", "--target", choices=targets_json.keys(), required=True, help="Sets the target, in which the shots are calculated. (Required)")
    day_parser.set_defaults(func=day_sub_command, gui=False)
    
    gui_parser = subparsers.add_parser("gui", help="Launches the program in gui mode", parents=[parent_parser])
    gui_parser.set_defaults(func=gui_sub_command, gui=True)
    
    # parser.add_argument("inputfile", help="sets the input file. Format delimited to .csv")
    # parser.add_argument("--output", "-o", default=None, help="sets the output file, defaults to <inputfile>_output.csv. Format delimited to .csv")
   


    global args
    args = parser.parse_args() if argv is None else parser.parse_args(argv)
    print(args)
    
    # LOGGING CONFIG
    if args.log is None: logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    else: logging.basicConfig(level=logging.WARNING, filename=args.log, filemode="w" ,format="%(asctime)s - %(levelname)s - %(message)s")
    global logger
    logger = logging.getLogger(__name__)
    if args.quiet: logger.setLevel(40)
    elif args.verbose: logger.setLevel(20)
    elif args.debug: logger.setLevel(10)   
    
    global CSV_DELIMETER
    CSV_DELIMETER = ";"
    global CALIBER
    CALIBER = args.caliber
    global RATIO
    RATIO = args.ratio   
    global TARGET_SET
    TARGET_SET = ""
    
    if not args.gui:
        if os.path.sep in args:
            outdir = os.path.dirname(args.output)
            if not os.path.exists(outdir): os.makedirs(outdir)
        
        if args.target is not None:
            set_target(args.target)
    
    args.func()
    
if __name__ == "__main__":
    main()
    