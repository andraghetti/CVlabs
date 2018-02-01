import numpy as np
import cv2

cap = cv2.VideoCapture(0)
edge_kernel = np.matrix('-1 -1 -1; -1 8 -1; -1 -1 -1')
sobel_kernel = np.matrix('1 2 1; 0 0 0; -1 -2 -1')
gaussian_kernel = np.matrix('1 2 1; 2 4 2; 1 2 1') / 32.0
T = 30

ready = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.flip(frame, 1, frame)

    if ready:
        # edge = cv2.filter2D(frame, -1, edge_kernel)
        noabs_diff = cv2.subtract(frame, previous) > T
        noabs_diff = (noabs_diff * 255).astype(np.uint8)

        cv2_absdiff = cv2.absdiff(frame, previous)

        chMask = np.abs(frame.astype(np.int32) - previous.astype(np.int32)) > T
        abs_diff = (chMask * 255).astype(np.uint8)
        cv2.imshow('Difference with abs', abs_diff)

        gaussian = cv2.filter2D(abs_diff, -1, gaussian_kernel)

        difference1 = cv2.threshold(cv2.absdiff(frame, previous), T, 255, cv2.THRESH_BINARY)[1]

        differenceA = cv2.absdiff(frame, previous) > T
        differenceB = cv2.absdiff(previous2, previous) > T
        difference3 = (np.bitwise_and(differenceA, differenceB) * 255).astype(np.uint8)

        previous2 = previous.copy()
        previous = frame.copy()

        # Display the resulting frame
        # cv2.imshow('Original', frame)
        #
        # cv2.imshow('Edge', edge)
        #cv2.imshow('Difference with cv2.subtract (no abs)', noabs_diff)
        #cv2.imshow('Difference with absdiff', cv2_absdiff)
        cv2.imshow('Difference with Threashold', difference1)
        cv2.imshow('Difference by 3', difference3)

        if cv2.waitKey(1) == 27:
            break
    else:
        previous = frame.copy()
        previous2 = previous.copy()
        ready = True

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()