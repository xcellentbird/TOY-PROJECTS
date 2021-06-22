import cv2
import time

cap = cv2.VideoCapture(0)

cnt = 0
prevtime = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    height = len(frame)
    width = len(frame[0])

    curtime = time.time()
    print(curtime - prevtime)

    if ret != 0:
        if cnt == 0:
            outimg = frame
        elif cnt < height-1:
            outimg[cnt] = frame[cnt]
            cv2.line(frame, (0, cnt+1), (width, cnt + 1), (255,0,0), 3)
            for r in range(height-1, cnt, -1):
                outimg[r] = frame[r]
        elif cnt == height:
            cv2.imwrite('outimg.png', outimg)
            break
            
        cv2.imshow('show', outimg)
        cnt+=1

        if cv2.waitKey(1) != -1:
            break
    else:
        break

img = cv2.imread('outimg.png')
cv2.imshow('outimg',img)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
    
