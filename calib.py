import numpy as np
import cv2
import functions as func

# Chessboard Parameters
cornerRow = 5
cornerCol = 8
patternSquareSide = 2.65

# for program
image_num = 8
n_cap = 10

cap = cv2.VideoCapture(0)

image_set = [0]*n_cap

while True:
    choice = input("Do you want to get new photos for calibration? (y|n)")
    if choice == 'y':
        i = 0
        ready = False
        print('Get 10 picture of the chessboard')

        while i < n_cap:
            ret, frame = cap.read()
            cv2.flip(frame, 1, frame)

            capture_window = 'Press c to capture, esc to exit'
            cv2.imshow(capture_window, frame)
            if ready:
                cv2.moveWindow(capture_window, 0, 100);

            preview = 'N°: ' + str(i + 1) + '. Press return to accept, c to retake'

            pressed = cv2.waitKey(1)
            if pressed == 27:  # esc
                break

            elif pressed == 99:  # C  (A=97, Z=122) or ord('c')
                cv2.imshow(preview, frame)
                cv2.moveWindow(preview, 650, 100)
                ready = True

            elif pressed == 10:  # return
                if ready:
                    image_set[i] = frame
                    cv2.imwrite('./data/cap' + str(i) + '.jpg', frame)
                    print('Captured and saved: ' + str(i+1))
                    cv2.destroyWindow(preview)
                    ready = False
                    i += 1

            else:
                if pressed != -1:
                    print("unknown key: " + str(pressed))
                continue
        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    elif choice == 'n':
        for j in range(n_cap):
            image_set[j] = cv2.imread('./data/cap'+str(j)+'.jpg')
        break
    else:
        continue

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cornerRow * cornerCol, 3), np.float32)
objp[:,:2] = np.mgrid[0:cornerCol, 0:cornerRow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


while True:
    choice = input("Do you want to see ALL the images? (y|n)")
    if choice == 'y':
        showall = True
        break
    elif choice == 'n':
        showall = False
        break

show_window = 'Corners found on chessboard'
# find chessboard corner
for img in image_set:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cornerCol, cornerRow), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # the window search size is 11 (2*5 + 1)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (cornerCol, cornerRow), corners2, ret)
        if showall:
            cv2.imshow(show_window, img)
            cv2.waitKey(0)

# show only one image
if not showall:
    cv2.imshow(show_window, image_set[image_num])
    cv2.waitKey(0)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n\n====Camera Parameters====\n")
print('A:\n' + str(np.round(mtx, 1)))
D = np.round(dist,3)[0].tolist()
print('\nDistortion (radial): ' + str(D[0]) + ', ' + str(D[1]) + ', ' + str(D[4]))
print('Distortion (tangent): ' + str(D[2]) + ', ' + str(D[3]))

# print('R:\n' + str(np.round(rvecs,1)))
# print('T:\n' + str(np.round(tvecs,1)))
cv2.destroyWindow(show_window)

img = cv2.imread('./data/cap'+str(image_num)+'.jpg')

print("\n\nLet's now find a corner by coordinates:")

coord_1p = [0,0]
while True:
    try:
        coord_1p[0] = int(input("Please enter x coord[0, " + str(cornerCol - 1) + "]:"))
        if coord_1p[0] >= cornerCol or coord_1p[0] < 0:
            print("Wrong")
            continue
        coord_1p[1] = int(input("Please enter y coord[0, " + str(cornerRow - 1) + "]:"))
        if coord_1p[1] >= cornerRow or coord_1p[1] < 0:
            print("Wrong")
            continue
    except ValueError:
        print("wrong: try again")
        continue
    else:
        break

green = (0, 255, 0)
pos = coord_1p[0] + coord_1p[1] * cornerCol
first_p = imgpoints[image_num][pos][0]
cv2.circle(img, tuple(first_p), 5, green, -1)  # -1 = filled

distance_win = 'Distance between points'
cv2.imshow(distance_win, img)
cv2.moveWindow(distance_win, 300, 600)

coord_2p = [0,0]

print("\n\nLet's now find a second one:")
while True:
    try:
        coord_2p[0] = int(input("Please enter x coord[0, " + str(cornerCol - 1) + "]:"))
        if coord_2p[0] >= cornerCol or coord_2p[0] < 0:
            print("Wrong")
            continue
        coord_2p[1] = int(input("Please enter y coord[0, " + str(cornerRow - 1) + "]:"))
        if coord_2p[1] >= cornerRow or coord_2p[1] < 0:
            print("Wrong")
            continue
    except ValueError:
        print("wrong: try again")
        continue
    else:
        break

red = (0, 0, 255)
pos = coord_2p[0] + coord_2p[1] * cornerCol
second_p = imgpoints[image_num][pos][0]
cv2.circle(img, tuple(second_p), 5, red, -1)  # -1 = filled
cv2.imshow(distance_win, img)


distance = round(np.sqrt(((pow(patternSquareSide*(coord_2p[0]-coord_1p[0]),2))+pow(patternSquareSide*(coord_2p[1]-coord_1p[1]),2))),2)
print("The distance between them is: " + str(distance) + "cm")

blue = (255,0,0)
cv2.line(img,tuple(first_p),tuple(second_p),blue,1)

rows,cols = img.shape[:2]
textImg = np.zeros(img.shape, img.dtype)

max = first_p if first_p[0]>second_p[0] else second_p
min = first_p if first_p[0]<second_p[0] else second_p

cv2.putText(textImg, str(distance) + "cm", tuple(min), cv2.FONT_HERSHEY_SIMPLEX, 1, blue,2)

angle = func.angle_between(min, max)
textImg = func.rotate_bound(textImg,angle)
img = func.addMeasure(img,textImg)

cv2.imshow(distance_win, img)
cv2.waitKey(0)

# END
cv2.destroyAllWindows()