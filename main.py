import csv
import argparse
from math import sqrt, atan, degrees
import json
import logging
import os
from alive_progress import alive_bar

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
    
def main(argv=None):
    parser = argparse.ArgumentParser(description="Meyton shot calculator and visualizer from coordinates")
    parser.add_argument("inputfile", help="sets the input file. Format delimited to .csv")
    parser.add_argument("--output", "-o", default=None, help="sets the output file, defaults to <inputfile>_output.csv. Format delimited to .csv")
    parser.add_argument("--caliber", "-c", default=4.5, type=int, help="sets the caliber in milimeters, defaults to standard 4,5mm (.177)")
    with open("targets.json", "r") as targets:
        targets_json = json.loads(targets.read())
        parser.add_argument("-t", "--target", required=True, choices=targets_json.keys(), help="sets the target, in which the shots are calculated (Required)")
    parser.add_argument("--ratio", default=100, help="defines the ratio beetween milimeters and units of distance in the meyton system (Default 1mm = 100 Unit of Distance)")

    qvd_parser = parser.add_mutually_exclusive_group()
    qvd_parser.add_argument("--quiet", "-q", action="store_true", help="Only shows errors and exceptions")
    qvd_parser.add_argument("--verbose", "-v", action="store_true", help="Shows the programs actions, warnings, errors and exceptions")
    qvd_parser.add_argument("--debug", action="store_true", help="Shows everything the program does. Only recommended for developers")
    parser.add_argument("--logfile", default=None, help="set the logfile name, defaults to stdout")

    args = parser.parse_args() if argv is None else parser.parse_args(argv)
    
    # LOGGING CONFIG
    if args.logfile is None: logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    else: logging.basicConfig(level=logging.WARNING, filename=args.logfile, filemode="w" ,format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    if args.quiet: logger.setLevel(40)
    elif args.verbose: logger.setLevel(20)
    elif args.debug: logger.setLevel(10)
    
    if args.output is None: outputname = f"{args.inputfile}_output.csv" 
    else: outputname = args.output
    
    if os.path.sep in outputname:
        outdir = os.path.dirname(outputname)
        if not os.path.exists(outdir): os.makedirs(outdir)
    
    global CALIBER
    CALIBER = args.caliber
    
    global RATIO
    RATIO = args.ratio
    
    CSV_DELIMETER = ";"
    
    target_data = targets_json[args.target]
    del targets_json
    target_data["rings"] = [(target_data["ten"] + target_data["quotient"] * i) / 2 for i in range(10)]

    target_data["ring_fraction"] = {float(f"10.{10 - i - 1}"): (target_data["ten"] / 10 * (i + 1)) / 2 for i in range(10)}
    target_data["ring_fraction"].update({(100 - i - 1) / 10: (target_data["ten"] + target_data["quotient"] / 10 * (i + 1)) / 2 for i in range(100)})
    # target_data["ring_fraction"].update({key: round(value, 2) for key, value in target_data["ring_fraction"].items()})
    
    logger.debug(f"RING FRACTION: {target_data['ring_fraction']}")
    
    global MEASURING_EDGE, MEASURING_RING
    MEASURING_EDGE = target_data["inner_ten_measuring_edge"]
    MEASURING_RING = target_data["inner_ten_measuring_ring"]

    logger.info("Initializing...")
    lines_in_file = open(args.inputfile, 'r').readlines()
    number_of_lines = len(lines_in_file)
    logger.info(f"INPUT FILE: {args.inputfile}")
    logger.info(f"OUTPUT FILE: {outputname}")
    with alive_bar(number_of_lines) as bar:
        # id    date    x   y   value   hypotenuse  angle
        # 0     1       2   3   4       5           6
        with open(args.inputfile, "r") as inputfile:
            with open(outputname, "a") as outputfile:
                csvreader = csv.reader(inputfile, delimiter=CSV_DELIMETER)
                for row in csvreader:
                    logger.debug(f"---NEW SHOT ID: {row[0]}---")
                    
                    hyp = getHypotenuse(row[2], row[3])
                    logger.debug(f"hypotenuse: {hyp}")
                
                    angle = getAngle(row[2], row[3])
                    logger.debug(f"angle: {angle}")

                    value = getValue(hyp, target_data["ring_fraction"])
                    logger.debug(f"value: {value}")
                    
                    outputfile.write(CSV_DELIMETER.join(map(str, (row[0], row[1], row[2], row[3], value, hyp, angle))) + "\n")
                    
                    if logger.level == 10:
                        if value == float(row[4].replace(",", ".")): logger.debug("TRUE: given value MATCHES the generated value")
                        else: logger.debug(f"FALSE ({value} != {row[4].replace(',', '.')}): given value DOESN'T MATCH the generated value")

                    bar()

if __name__ == "__main__":
    main()
    