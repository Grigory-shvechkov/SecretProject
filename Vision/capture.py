import cv2

class Camera:
    def __init__(self,index=0,width=640,height=480):
        self.index = index
        self.cap = cv2.VideoCapture(index,cv2.CAP_V4L2)

        if not self.cap.isOpened():
            # Don't crash the whole pipeline over one missing/unplugged
            # camera -- print why and leave self.cap as None so read()
            # just reports "no frame" forever, same as a dropped frame.
            print(f"Warning: Camera {index} could not be opened. "
                  f"Continuing without it -- frames from it will be None.")
            self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def read(self):
        if self.cap is None:
            return None
        ok,frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    #these dunders are for the with x, as y blocks (safe clean up)

    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.release()

    