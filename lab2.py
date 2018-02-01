import numpy as np
import cv2
import functions as lab2

# # the histogram of the data
# plt.figure(3)
# plt.hist(hist, bins=256, range=[0,256], facecolor='green', histtype='bar',rwidth=50)
# plt.xlabel('Greylevels i ')
# plt.ylabel('h(i)')
# plt.title('Histogram of ')
# plt.grid(True)
# plt.show()

while True:
    choice = int(input(
        'What do you want to do: 1)linear contrast stretch 2)gamma correction 3)equalization 4)convolution 5)exit'))
    if choice == 1:
        img = cv2.imread('./data/wom1.bmp', cv2.IMREAD_GRAYSCALE)
        imge = lab2.linear_contrast_stretch(img)
        lab2.show(img, imge)
    elif choice == 2:
        img = cv2.imread('./data/fce4.bmp', cv2.IMREAD_GRAYSCALE)
        imge = lab2.gamma_correction(img, 1.5)
        lab2.show(img, imge)
    elif choice == 3:
        img = cv2.imread('./data/pum1dim1.bmp', cv2.IMREAD_GRAYSCALE)
        imge = lab2.equalization(img)
        lab2.show(img, imge)
    elif choice == 4:
        img = cv2.imread('./data//fce5moregaussnoise.bmp', cv2.IMREAD_GRAYSCALE)
        kernel = [[1 / 9.0, 2 / 9.0, 1 / 9.0],
                  [2 / 9.0, 4 / 9.0, 2 / 9.0],
                  [1 / 9.0, 2 / 9.0, 1 / 9.0]]
        k = int(np.sqrt(len(kernel)) - 1 / 2)
        imge = lab2.convolution(img, kernel, k)
        lab2.show(img, imge)
    elif choice == 5:
        break
