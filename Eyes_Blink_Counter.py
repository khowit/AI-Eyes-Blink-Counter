import cv2
import Face_MeshModule as fm
from cvzone.PlotModule import LivePlot
import cvzone


# cap = cv2.VideoCapture("image/blink.mp4")
cap = cv2.VideoCapture(0)
detector = fm.FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[0,50],invert=True)

idList = [22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []
BlinkCounter = 0
counter = 0
color = (255,0,255)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES)

    success, frame = cap.read()
    frame, faces = detector.findFaceMesh(frame, draw=False)
    

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(frame, face[id], 5, color, cv2.FILLED)

        leftup = face[159]
        leftdown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer,_ = detector.findDistance(leftup, leftdown)
        lengthHor,_ = detector.findDistance(leftLeft, leftRight)
        cv2.line(frame, leftup, leftdown, (0,255,0), 3)
        cv2.line(frame, leftLeft, leftRight, (0,255,0), 3)
        # print(length)

        ratio = int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)

        if len(ratioList)>3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 30 and counter == 0:
            BlinkCounter += 1
            color = (0,255,0)
            counter = 1
            

        if counter != 0:
            counter += 1
            if counter > 13:
                counter = 0
                color = (255,0,255)

        cvzone.putTextRect(frame, f'Blink Count : {BlinkCounter}',(50,100),colorR=color)
        framePlot = plotY.update(ratioAvg, colors=color)
        frame = cv2.resize(frame, (640,360))
        frameStack = cvzone.stackImages([frame,framePlot],2,1)
    else:
        frame = cv2.resize(frame, (640,360))
        frameStack = cvzone.stackImages([frame,frame],2,1)

    # frame = cv2.resize(frame, (640,360))
    cv2.imshow("Output", frameStack)
    if cv2.waitKey(25) & 0xFF == ord("e"):
            break 

cap.release()
cv2.destroyAllWindows()