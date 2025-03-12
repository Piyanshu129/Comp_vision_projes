from ultralytics import YOLO
import cv2
import cvzone
import math
# cap =cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)

cap=cv2.VideoCapture("../PPE-trained-model/ppe2.mp4")
#For videos

model=YOLO("../PPE-trained-model/best (1).pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

while True:
    success,img=cap.read()
    results=model(img,stream=True)

    for r in results:
        boxes=r.boxes
        for box in boxes:


            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
             # bbox=int(x1) ,int(y1) , int(w), int(h)



#bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
#confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

#classsname
            cls=int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(30, y1)))

    cv2.imshow("Image",img)
    cv2.waitKey(1)


