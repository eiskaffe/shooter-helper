import pytesseract
import PIL.Image
import cv2
import csv
import argparse
from math import sqrt, atan, degrees


def hypotenuse(x, y):
    return sqrt(int(x) ** 2 + int(y) ** 2)

def getAngle(x, y):
    x = int(x)
    y = int(y)
    
    q = None    # Quadrants
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

def getValue():
    return
    
def main(argv=None):
    parser = argparse.ArgumentParser(description="Meyton shot calculator and visualizer from coordinates")
    parser.add_argument("inputfile", help="sets the input file")
    parser.add_argument("--output", "-o", help="sets the output file, defaults to <inputfile>_output.csv")
    parser.add_argument("--caliber", "-c", default=4.5, type=int, help="sets the caliber in milimeters, defaults to standard 4,5mm (.177)")
    
    args = parser.parse_args() if argv is None else parser.parse_args(argv)
    caliber = args.caliber

    # id    date    x   y   value   hypotenuse  angle
    # 0     1       2   3   4       5           6
    with open(args.inputfile, newline='') as inputfile:
        csvreader = csv.reader(inputfile, delimiter=';')
        for row in csvreader:
            try: print(hypotenuse(row[2], row[3]), end=" ")
            except ValueError: pass
            
            try: print(getAngle(row[2], row[3]))
            except ValueError: pass
            

if __name__ == "__main__":
    main()