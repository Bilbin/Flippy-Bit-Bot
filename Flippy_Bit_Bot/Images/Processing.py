import cv2

a = cv2.imread("processing.png")

b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

lowerGray = 133
upperGray = 255
c = cv2.inRange(b, lowerGray, upperGray)

cv2.imwrite("processing2.png", c)