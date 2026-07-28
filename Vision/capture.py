import cv2

class Camera:
    def __init__(self,index=0,width=640,height=480):
        self.index = index
        self.cap = cv2.VideoCapture(index,cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise Exception(f"Camera {index} could not be opened.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def read(self):
        ok,frame = self.cap.read()
        return frame if ok else None
    
    def release(self):
        self.cap.release()

    #these dunders are for the with x, as y blocks (safe clean up)

    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.release()

    