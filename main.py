# Math
from email.mime import base
from math import sqrt, atan, degrees, floor
import numpy as np
# Image processing
import imglibrary
import cv2
import pytesseract
from pytesseract import Output
# Data management
import json
import csv
# Other
import logging
import os
from alive_progress import alive_bar
import argparse
import datetime

# METADATA
VERSION = 0.1
VERSION_TYPE = "development"
AUTHOR = "eiskaffe"

def getHypotenuse(x, y):
    return sqrt(int(x) ** 2 + int(y) ** 2)

def getAngle(x, y):
    x = int(x)
    y = int(y)
    
    q = None
    if x > 0 and y >= 0: q = 1
    elif x <= 0 and y > 0: q = 2
    elif x < 0 and y <= 0: q = 3
    elif x >= 0 and y < 0: q = 4
    
    if x == 0 or y == 0:
        if x == 0 and y == 0: return 0
        else: return (q - 1) * 90
    else:
        deg = degrees(abs(atan(y / x)))
        if q == 1: return deg
        else: return (q * 90) - deg

def getValue(hyp, ring_fraction):
    distance = hyp / RATIO
    if innerTen(distance, ring_fraction) is False: distance = distance - CALIBER / 2
    for key, item in ring_fraction.items():
        if distance <= item: return key
    return None

def innerTen(hyp, rings):
    if hyp + MEASURING_EDGE / 2 <= rings[MEASURING_RING]: return True
    else: return False

def longestStringIn2DList(inputlist):
    # assuming all rows are the same length
    columns = len(inputlist[0])
    column_length = [0 for _ in range(columns)]
    for row in inputlist:
        for i, value in enumerate(row):
            if len(value) > column_length[i]: column_length[i] = len(value)
    return column_length

def printTable(list2d:list[list], head:list=[], style="|-+", firstlinehead=True):
    # STYLE: 0: vertical, 1: horizontal, 2: intersection
    head = [head]
    column_length = longestStringIn2DList(list2d + head)
    style = [*style]
    splitter = style[2]
    for value in column_length:
        splitter = f"{splitter}{style[1]*(value + 2)}{style[2]}"
    print(splitter)
        
    if head != [[]]:
        print_row = style[0]
        for column, max_length in zip(head[0], column_length):
            print_row = f"{print_row} {column.upper()}{' '*(max_length - len(column) + 1)}{style[0]}"
        print(print_row)
        print(splitter)
        
    for row in list2d:
        print_row = style[0]
        for column, max_length in zip(row, column_length):
            print_row = f"{print_row} {column}{' '*(max_length - len(column) + 1)}{style[0]}"
        print(print_row)
    print(splitter)
    
def manualTenInsertion():
    ...
     
def imageParser(img_path, allowed_characters="-0123456789", config = r"--psm 12"):
    if os.path.exists(img_path) is False: raise FileNotFoundError(f"Given image file ({img_path}) does not exist.")
    img = cv2.imread(img_path)
    img = imglibrary.noiseRemoval(img)
    # img = imglibrary.deskew(img)
    img = imglibrary.grayscale(img)
    thresh, img = cv2.threshold(img, 100, 300, cv2.THRESH_BINARY)
    img = imglibrary.thickFont(img, 15)
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
    logger.debug(f"FILTERED LIST {filtered_list}")
    
    final_list = filtered_list.copy()
    while len([word for word in final_list if all(letter in allowed_characters for letter in word)]) != 20:
        printTable([[str(i), str(val), f"{conf}% {' !' if conf < 80 else ''}"] for i, (val, conf) in enumerate(zip(filtered_list, confidence))], head=["id", "value", "confidence"])
        input("Please select an ID that needs to be edited.")
    
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

    # return [[final_list[i], final_list[i+1]] for i in range(0, len(final_list), 2)]

def draw(shots, target, font_size = 25):
    img = np.zeros((target["card_size"][0]*RATIO, target["card_size"][1]*RATIO, 3), np.uint8)
    img[::] = (192, 21, 0)
    center = (target["card_size"][0] // 2 * RATIO, target["card_size"][1] // 2 * RATIO)
    xmid, ymid = center
    quotient = int(target["quotient"]*RATIO) // 2
    ring_thickness = int(target["ring_thickness"]*RATIO)
    img = cv2.circle(img, center, 0, (255, 255, 255), int(target["rings"][-1]*2*RATIO))
    img = cv2.circle(img, center, 0, (90, 144, 21), int(target["ring_fraction"][target["black"]]*2*RATIO))
    if logger.level == 10:
        img = cv2.circle(img, center, 0, (0, 0, 255), 200)
        img = cv2.line(img, (xmid, 0), (xmid, target["card_size"][1]*RATIO), (0, 0, 255), 50)
        img = cv2.line(img, (0, ymid), (target["card_size"][0]*RATIO, ymid), (0, 0, 255), 50)
    for i, radius in enumerate(target["rings"]):
        if 10 - i < target["black"]: color = (0, 0, 0)
        else: color = (235, 235, 235)
        img = cv2.circle(img, center, int(radius*RATIO), color, ring_thickness)
        
        if 10 - i <= target["numbers"][0] and 10 - i >= target["numbers"][1]:
            (label_width, label_height), baseline = cv2.getTextSize(str(10-i), cv2.FONT_HERSHEY_PLAIN, font_size, font_size*2)
            
            img = cv2.putText(img, str(10-i), (xmid - font_size*RATIO//20, ymid - i*quotient), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
            img = cv2.putText(img, str(10-i), (xmid - font_size*RATIO//20, ymid + i*quotient + label_height + label_height//8), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
            
            img = cv2.putText(img, str(10-i), (xmid - i*quotient - label_width + label_width//8, ymid + font_size*RATIO//20), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
            img = cv2.putText(img, str(10-i), (xmid + i*quotient, ymid + font_size*RATIO//20), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_size*2)
        
    for shot in shots:
        x = int(shot[2])
        y = int(shot[3])
        xmid = x + target["card_size"][0]//2*RATIO if x >= 0 else target["card_size"][0]//2*RATIO - abs(x)
        ymid = target["card_size"][0]//2*RATIO - y if y >= 0 else abs(y) + target["card_size"][0]//2*RATIO
        img = cv2.circle(img, (xmid, ymid), 0, (0, 0, 0), int(CALIBER*RATIO))
        img = cv2.circle(img, (xmid, ymid), int(CALIBER*RATIO)//2, (255, 255, 255), int(target["ring_thickness"]*RATIO))
    
    
    
    
    
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1000, 1000)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    
def evaluateDay(lst):
    ...
    
# SUB-COMMANDS
def imageSubCommand():
    image_data = [imageParser(img_path) for img_path in args.imagefiles]
    if args.target is not None:
        lst = [[f"{datetime.datetime.now().strftime('%y%m%d')}_{(i1*10) + i2 + 1}", 
                f"{datetime.datetime.now().strftime('%Y.%m.%d')}",
                shot[0], 
                shot[1],
                getValue(getHypotenuse(shot[0], shot[1]), target_data["ring_fraction"]),
                getAngle(shot[0], shot[1]),
                getHypotenuse(shot[0], shot[1])
                ] for i1, tens in enumerate(image_data) for i2, shot in enumerate(tens)]
    else:
        lst = [[f"{datetime.datetime.now().strftime('%y%m%d')}_{(i1*10) + i2 + 1}", 
                f"{datetime.datetime.now().strftime('%Y.%m.%d')}",
                shot[0], 
                shot[1],
                None,
                getAngle(shot[0], shot[1]),
                getHypotenuse(shot[0], shot[1])
                ] for i1, tens in enumerate(image_data) for i2, shot in enumerate(tens)]
    
    if args.debug:
        printTable([list(map(str, tens)) for tens in lst], head=["id", "date", "x", "y", "value", "angle", "hyp"], firstlinehead=False)
        
    if args.target is not None: draw(lst, target_data)
    
    with open(args.output, "a") as outfile:
        for shot in [list(map(str, tens)) for tens in lst]:
            outfile.write(CSV_DELIMETER.join(shot) + "\n")
      
def daySubCommand():
    with open(args.csvfile, "r") as infile:
        csvreader = csv.reader(infile, delimiter=CSV_DELIMETER)
        # implementing linear search
        # id    date    x   y   value   hypotenuse  angle
        # 0     1       2   3   4       (5)         (6)
        shots = [row for row in csvreader if row[1] == args.date]
    
    print(shots)
    draw(shots, target_data)
                
            
    
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
    # Other
    parser = argparse.ArgumentParser(prog="Shooter Helper", description="Meyton shot calculator and visualizer from coordinates")
    parser.add_argument("--version", action="version", version=f"shooterhelper {VERSION} {VERSION_TYPE} by {AUTHOR}", help="Shows the version")

    # Sub-commands
    subparsers = parser.add_subparsers(help="Sub-commands")
    
    img_parser = subparsers.add_parser("image", help="Image input related", parents=[parent_parser])
    img_parser.add_argument("imagefiles", nargs="*", help="Reads the shots' coordinates from the images. The images order are set by their names. ")
    img_parser.add_argument("--output", "-o", default=f"{date}-shots.csv", help=f"Sets the output file. File format delimited to csv. Defaults to {date}-shots.csv")
    with open("targets.json", "r") as targets:
            targets_json = json.loads(targets.read())
            img_parser.add_argument("-t", "--target", choices=targets_json.keys(), help="Sets the target, in which the shots are calculated. If not set the program will not calculate the shots' value (and others)!")
    img_parser.add_argument("--disable-errors", action="store_true", help="When this argument is given, the program terminates, and will not try to get user input to solve the issue")
    # img_parser.add_argument("--print-table", "-p", type=bool, default=None) 
    img_parser.set_defaults(func=imageSubCommand)
    
    csv_parser = subparsers.add_parser("calculate", help="lorem ipsum", parents=[parent_parser])
    
    day_parser = subparsers.add_parser("day", help="lorem ipsum", parents=[parent_parser])
    day_parser.add_argument("csvfile", help="Reads this file for data")
    day_parser.add_argument("date", nargs="?", default=datetime.datetime.now().strftime('%Y.%m.%d'), help=f"Set the date to parse, defaults to today's date as ({datetime.datetime.now().strftime('%Y.%m.%d')})")
    with open("targets.json", "r") as targets:
        targets_json = json.loads(targets.read())
        day_parser.add_argument("-t", "--target", choices=targets_json.keys(), required=True, help="Sets the target, in which the shots are calculated. (Required!)")
    day_parser.set_defaults(func=daySubCommand)
    
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
    
    if os.path.sep in args:
        outdir = os.path.dirname(args.output)
        if not os.path.exists(outdir): os.makedirs(outdir)
    
    global CALIBER
    CALIBER = args.caliber
    
    global RATIO
    RATIO = args.ratio
    
    global CSV_DELIMETER
    CSV_DELIMETER = ";"
    
    global target_data
    target_data = []
    if args.target is not None:
        target_data = targets_json[args.target]
        logger.debug(f"TARGET DATA: {target_data}")
        del targets_json
        target_data["rings"] = [(target_data["ten"] + target_data["quotient"] * i) / 2 for i in range(10)]
        logger.debug(f"RINGS: {target_data['rings']}")

        target_data["ring_fraction"] = {float(f"10.{10 - i - 1}"): (target_data["ten"] / 10 * (i + 1)) / 2 for i in range(10)}
        target_data["ring_fraction"].update({(100 - i - 1) / 10: (target_data["ten"] + target_data["quotient"] / 10 * (i + 1)) / 2 for i in range(100)})
        target_data["ring_fraction"].update({key: round(value, 2) for key, value in target_data["ring_fraction"].items()})
        
        logger.debug(f"RING FRACTION: {target_data['ring_fraction']}")
        
        global MEASURING_EDGE, MEASURING_RING
        MEASURING_EDGE = target_data["inner_ten_measuring_edge"]
        MEASURING_RING = target_data["inner_ten_measuring_ring"]
        
    args.func()

    # logger.info("Initializing...")
    # lines_in_file = open(args.inputfile, 'r').readlines()
    # number_of_lines = len(lines_in_file)
    # logger.info(f"INPUT FILE: {args.inputfile}")
    # logger.info(f"OUTPUT FILE: {args.output}")
    # with alive_bar(number_of_lines) as bar:
    #     # id    date    x   y   value   hypotenuse  angle
    #     # 0     1       2   3   4       (5)         (6)
    #     with open(args.inputfile, "r") as inputfile:
    #         with open(args.output, "a") as outputfile:
    #             csvreader = csv.reader(inputfile, delimiter=CSV_DELIMETER)
    #             for row in csvreader:
    #                 logger.debug(f"---NEW SHOT ID: {row[0]}---")
    #                 if row[2] == "" or row[3] == "": 
    #                     print(row)
    #                     logger.warning(f"SHOT ID {row[0]} has incomplete coordinates (X:{'None' if row[2] == '' else row[2]};Y:{'None' if row[3] == '' else row[3]})")
    #                     outputfile.write(CSV_DELIMETER.join(map(str, (row[0], row[1], "" if row[2] == "" else row[2], "" if row[3] == "" else row[3], "" if row[4] == "" else row[4]))) + "\n")
    #                 else:
    #                     hyp = getHypotenuse(row[2], row[3])
    #                     logger.debug(f"hypotenuse: {hyp}")
                    
    #                     angle = getAngle(row[2], row[3])
    #                     logger.debug(f"angle: {angle}")

    #                     value = getValue(hyp, target_data["ring_fraction"])
    #                     logger.debug(f"value: {value}")
                        
    #                     if logger.level == 10:
    #                         if value == float(row[4].replace(",", ".")): logger.debug("TRUE: given value MATCHES the generated value")
    #                         else: logger.debug(f"FALSE ({value} != {row[4].replace(',', '.')}): given value DOESN'T MATCH the generated value")
                    
    #                     outputfile.write(CSV_DELIMETER.join(map(str, (row[0], row[1], row[2], row[3], value, hyp, angle))) + "\n")

    #                 bar()

if __name__ == "__main__":
    main()
    