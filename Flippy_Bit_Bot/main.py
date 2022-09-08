import PIL.ImageGrab
import cv2
import numpy as np
from PIL import Image
from pynput.keyboard import Listener
import matplotlib.pyplot as plt
from queue import Queue
import itertools
import math
import pyautogui
import time
import os
import threading

IM_WIDTH = 606
IM_HEIGHT = 755
CONFIDENCE_THRESHOLD = 0.89
POINT_CONSOLIDATION_RADIUS = 2
PAIRING_HEIGHT_DIFFER_ALLOWANCE = 2
FIRST_BIT_X = 672
FIRST_BIT_Y = 905
BIT_GAP = 77
BIT_ON_RGB = (255, 255, 255)
BIT_OFF_RGB = (34, 17, 34)
EMPTY_BYTE = "00000000"
CHECK_INTERVAL = 0.5

threadQueue = Queue()

pyautogui.PAUSE = 0.02


bitLocations = [(FIRST_BIT_X + (i * BIT_GAP), FIRST_BIT_Y) for i in range(8)]
bitStates = ["0" for i in range(8)]


def main():
    mask = getMask()

    allGuesses = getGuesses(mask)

    # threadQueue.put((plot, allGuesses))

    guessPoints = consolidatePoints(allGuesses)

    guessPointPairs = getGuessPointPairs(guessPoints)

    binBytes = getBytes(guessPointPairs)
    print(binBytes)

    global bitStates
    for pairGuessList in binBytes:
        for guess in pairGuessList:
            for index, digit in enumerate(guess):
                if bitStates[index] != digit:
                    # Flip the bit in game and in the bit states array
                    flipBit(index)
                    bitStates[index] = "1" if bitStates[index] == "0" else "0"

            time.sleep(0.05)
            im = PIL.ImageGrab.grab(bbox=(FIRST_BIT_X, FIRST_BIT_Y, FIRST_BIT_X + (BIT_GAP * 8), FIRST_BIT_Y + 1))
            realBitState = getRealBitState(im.load())

            bitStates = list(realBitState)

            if realBitState == EMPTY_BYTE:
                break


def schedule():
    while True:
        main()
        time.sleep(CHECK_INTERVAL)


def getRealBitState(image):
    realBitState = ""
    for i in range(8):
        pixel = image[i * BIT_GAP, 0]
        if pixel == BIT_ON_RGB:
            realBitState += "1"
        else:
            realBitState += "0"

    return realBitState


def getBytes(guessPointPairs):
    binBytes = []

    for pair in guessPointPairs:
        pairGuesses = []

        for hexString in pair.guesses:
            hexNum = int(hexString, 16)

            binary = f'{hexNum:0>8b}'
            binaryString = str(binary)
            pairGuesses.append(binaryString)

        binBytes.append(pairGuesses)

    return binBytes


def flipBit(index):
    loc = bitLocations[index]
    pyautogui.moveTo(loc[0], loc[1])
    pyautogui.drag(40, 0, 0.001)


def getGuessPointPairs(guessPoints):

    pairs = []

    # Used to check if loop should still be ran (points are still being joined)
    pairJoined = True

    while pairJoined:

        pairJoined = False

        for guessPoint1, guessPoint2 in itertools.combinations(guessPoints, 2):
            if math.fabs(guessPoint2.y - guessPoint1.y) < PAIRING_HEIGHT_DIFFER_ALLOWANCE:
                guessPoints.remove(guessPoint1)
                guessPoints.remove(guessPoint2)

                if guessPoint2.x > guessPoint1.x:
                    pair = GuessPointPair(guessPoint1, guessPoint2)
                else:
                    pair = GuessPointPair(guessPoint2, guessPoint1)

                pairs.append(pair)

                pairJoined = True
                break

    # Dummy guess point for hex bytes that only have one digit (last four bits.) Represents leading 0
    dummyGuessPoint = generateDummyGuessPoint()

    for remainingGuessPoint in guessPoints:
        lonePair = GuessPointPair(dummyGuessPoint, remainingGuessPoint)
        pairs.append(lonePair)

    for pair in pairs:
        pairGuesses = []

        for pairGuess in [x.digit + y.digit for x in pair.leftPoint.guesses for y in pair.rightPoint.guesses]:
            pairGuesses.append(pairGuess)

        pairGuessSet = set(pairGuesses)
        pairGuesses = list(pairGuessSet)
        pair.guesses = pairGuesses

    return pairs


def generateDummyGuessPoint():
    dummyGuessPoint = GuessPoint(0, 0)
    dummyGuess = Guess("0", 0, 0, 1.0)
    dummyGuessPoint.guesses = [dummyGuess]

    return dummyGuessPoint


# Takes guesses and changes their coordinates to their guess point (average of points that are meant for the same digit)
def consolidatePoints(guesses):
    guessPoints = getGuessPoints(guesses)

    # Find guess point with minimum distance and add guess to guess point list
    for guess in guesses:
        # Dummy values so first evaluation will stick
        minDistAndPoint = (math.inf, guessPoints[0])

        for guessPoint in guessPoints:
            distance = math.dist((guess.x, guess.y), (guessPoint.x, guessPoint.y))

            if distance < minDistAndPoint[0]:
                minDistAndPoint = (distance, guessPoint)

        minDistAndPoint[1].guesses.append(guess)

    # Sort guesses for each guess point in order of confidence
    for guessPoint in guessPoints:
        guessPoint.guesses = sorted(guessPoint.guesses, key=lambda x: x.confidence, reverse=True)

    return guessPoints


def getGuessPoints(guesses):
    points = []
    for guess in guesses:
        points.append((guess.x, guess.y))

    # Used to check if loop should still be ran (points are still being joined)
    pairJoined = True

    while pairJoined:

        pairJoined = False

        for point1, point2 in itertools.combinations(points, 2):
            if math.dist(point1, point2) < POINT_CONSOLIDATION_RADIUS:
                points.remove(point1)
                points.remove(point2)

                avgX = (point1[0] + point2[0]) / 2
                avgY = (point1[1] + point2[1]) / 2

                points.append((avgX, avgY))
                pairJoined = True
                break

    guessPoints = []

    for point in points:
        guessPoint = GuessPoint(point[0], point[1])
        guessPoints.append(guessPoint)

    return guessPoints


def plot(guesses):
    plt.xlim(0, IM_WIDTH)
    plt.ylim(0, IM_HEIGHT)
    plt.gca().set_aspect('equal', adjustable='box')

    X = []
    Y = []

    for guess in guesses:
        X.append(guess.x)
        Y.append(IM_HEIGHT - guess.y)
        plt.annotate(guess.digit, (guess.x, IM_HEIGHT - guess.y))

    plt.scatter(X, Y)

    plt.show()


# Describes a guess from template matching within a certain confidence threshold
class Guess:
    def __init__(self, digit, x, y, confidence):
        self.digit = digit
        self.x = x
        self.y = y
        self.confidence = confidence


# A point that consolidates nearby coordinates of guesses
class GuessPoint:
    def __init__(self, x, y):
        self.guesses = []
        self.x = x
        self.y = y


# A pair of guess points (makes up a full hex byte)
class GuessPointPair:
    def __init__(self, leftPoint, rightPoint):
        self.leftPoint = leftPoint
        self.rightPoint = rightPoint
        self.guesses = []


def getGuesses(mask):
    allGuesses = []

    for file in os.listdir("Images"):
        if "png" in file:
            matchImage = cv2.imread(os.path.join("Images", file))
            matchImage = cv2.cvtColor(matchImage, cv2.COLOR_RGB2GRAY)
            matches = cv2.matchTemplate(mask, matchImage, cv2.TM_CCORR_NORMED)

            while cv2.minMaxLoc(matches)[1] > CONFIDENCE_THRESHOLD:
                match = cv2.minMaxLoc(matches)
                row = match[3][0]
                column = match[3][1]
                guess = Guess(file[0], row, column, match[1])
                allGuesses.append(guess)
                matches[column, row] = 0

    return allGuesses


def getMask():
    image = PIL.ImageGrab.grab(bbox=(656, 52, 1262, 807))
    # image = PIL.Image.open("test.png")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lowerGray = 133
    upperGray = 255
    mask = cv2.inRange(image, lowerGray, upperGray)
    im = PIL.Image.fromarray(mask)
    # im.show()
    return mask


def onPress(key):
    try:
        if key.char == '[':
            threading.Thread(target=schedule).start()

    except AttributeError:
        pass


def listenForPlot():
    plotDetails = threadQueue.get()
    plotDetails[0](plotDetails[1])


# Create keyboard listener on different thread
listener = Listener(on_press=onPress)
listener.start()

# On main thread, listen for the request to plot
listenForPlot()
