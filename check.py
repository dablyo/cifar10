import cv2
img = cv2.open('/home/wang/git/cifar10/pic/0-1__.jpg')
img = img.resize((28,28),0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
