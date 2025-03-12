from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with available parameters
detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)

# Continuously get frames from the webcam
while True:
    success, img = cap.read()

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1['center']
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")

        # Corrected line: Removed 'color' argument
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)
            print(fingers2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Corrected line: Removed 'color' argument
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)

        print(" ")  # New line for better readability of the printed output

    # Display the image in a window
    cv2.imshow("Image", img)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)
