import cv2
import mediapipe as mp
import numpy as np
import time
import math

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, findLandmark=True, minDetectioncon=0.5, minTrackcon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.findLandmark = findLandmark
        self.minDetectioncon = minDetectioncon
        self.minTrackcon = minTrackcon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.findLandmark, self.minDetectioncon, self.minTrackcon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0,0,255))

    def findFaceMesh(self, frame, draw=True):
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                faces.append(face)

        return frame, faces

    def findDistance(self,p1, p2, img=None):   
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture("image/musk.mp4")
    pTime = 0
    cTime = 0
    detector = FaceMeshDetector()
    while True:
        success, frame = cap.read()
        frame, faces = detector.findFaceMesh(frame)
        if len(faces) != 0:
            print(len(faces[0]))


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f'FPS:{int(fps)}', (10,150),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break 

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
