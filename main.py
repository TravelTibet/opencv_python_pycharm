# import cv2
# cv2.imshow("Tom",cv2.imread("./pic/tom.png"))
# retval = cv2.waitKey()
#

# import cv2
# lena=cv2.imread("./pic/tom.png")
# cv2.imshow("demo", lena )
# key=cv2.waitKey()
# if key==ord('A'):
#  cv2.imshow("PressA",lena)
# elif key==ord('B'):
#  cv2.imshow("PressB",lena)

# import cv2
# lena=cv2.imread("./pic/tom.png")
# cv2.imshow("demo", lena )
# key=cv2.waitKey()
# if key!=-1:
#     print("触发了按键")

# import cv2
# pic = cv2.imread("./pic/tom.png")
# cv2.imshow("tom",pic)
# cv2.waitKey()
# cv2.destroyWindow("demo")

# import cv2
# import numpy as np
# img = np.zeros((8,8),dtype = np.uint8)
# print("img = \n",img)
# cv2.imshow("one",img)
# print("读取像素点 img [0,3] = ",img[0,3])
# img[0,3] = 255
# print("修改后 img=\n",img)
# print("读取修改后的像素点 img [0,3]= " ,img[0,3])
# cv2.imshow("two",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img = cv2.imread("./pic/lena.bmp",0)
# cv2.imshow("before",img)
# for i in range(10,100):
#     for j in range(80,100):
#         img[i,j] = 255
# cv2.imshow("after",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# #-----------蓝色通道值--------------
# blue=np.zeros((300,300,3),dtype=np.uint8)
# blue[:,:,0]=255
# print("blue=\n",blue)
# cv2.imshow("blue",blue)
# #-----------绿色通道值--------------
# green=np.zeros((300,300,3),dtype=np.uint8)
# green[:,:,1]=255
# print("green=\n",green)
# cv2.imshow("green",green)
# #-----------红色通道值--------------
# red=np.zeros((300,300,3),dtype=np.uint8)
# red[:,:,2]=255
# print("red=\n",red)
# cv2.imshow("red",red)
# #-----------释放窗口--------------
# cv2.waitKey()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# img = np.zeros((300,300,3),dtype = np.uint8)
# img[:,0:100,0] = 255 # 蓝色
# img[:,100:200,1] = 255 # 绿色
# img[:,200:300,2] = 255 # 红色
# print("img = \n",img)
# cv2.imshow("img",img)
# cv2.waitKey()
# # cv2.destroyAllWindows()

# import numpy as np
# import cv2
# img = np.zeros((2, 4, 3), dtype=np.uint8)
# print("img=\n", img)
# cv2.imshow("before",img)
# print("读取像素点 img[0,3]=", img[0, 3])
# print("读取像素点 img[1,2,2]=", img[1, 2, 2])
# img[0, 3] = 255
# img[0, 0] = [66, 77, 88]
# img[1, 1, 1] = 3
# img[1, 2, 2] = 4
# img[0, 2, 0] = 5
# print("修改后 img\n", img)
# print("读取修改后像素点 img[1,2,2]=", img[1, 2, 2])
# cv2.imshow("after",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img = cv2.imread("./pic/Lena.bmp")
# cv2.imshow("before", img)
# print("访问 img[0,0]=", img[0, 0])
# print("访问 img[0,0,0]=", img[0, 0, 0])
# print("访问 img[0,0,1]=", img[0, 0, 1])
# print("访问 img[0,0,2]=", img[0, 0, 2])
# print("访问 img[50,0]=", img[50, 0])
# print("访问 img[100,0]=", img[100, 0])
# # for i in range(0, 50):
# #     for j in range(0, 100):
# #         for k in range(0, 3):
# #             img[i, j, k] = 255 # 三通道都是 255，纯白色
# for i in range(0, 50):
#     for j in range(0, 100):
#             img[i, j] = 255 # 三通道都是 255，纯白色
#
# for i in range(50, 100):
#     for j in range(0, 100):
#         img[i, j] = [128, 128, 128]  # 灰色
#
# for i in range(100,150):
#     for j in range(0,100):
#         img[i,j] = 0 # 三通道都是 0，纯黑色
# cv2.imshow("after",img)
# print("修改后 img[0,0]=",img[0,0])
# print("修改后 img[0,0,0]=",img[0,0,0])
# print("修改后 img[0,0,1]=",img[0,0,1])
# print("修改后 img[0,0,2]=",img[0,0,2])
# print("修改后 img[50,0]=",img[50,0])
# print("修改后 img[100,0]=",img[100,0])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
# img=np.random.randint(10,199,size=[50,50],dtype=np.uint8)
# cv2.imshow("before",img)
# print("img=\n",img)
# print("读取像素点 img.item(3,2)=",img.item(3,2))
# # img.itemset((3,2),255)
# img[3,2] = 255
# cv2.imshow("after",img)
# print("修改后 img=\n",img)
# print("修改后像素点 img.item(3,2)=",img.item(3,2))
# cv2.waitKey()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# img=np.random.randint(0,256,size=[512,512,3],dtype=np.uint8)
# cv2.imshow("demo",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
img=cv2.imread("./pic/Lena.bmp",0)
#测试读取、修改单个像素值
print("读取像素点 img.item(3,2)=",img.item(3,2))
# img.itemset((3,2),255)
img[3,2] = 255
print("修改后像素点 img.item(3,2)=",img.item(3,2))
#测试修改一个区域的像素值
cv2.imshow("before",img)
for i in range(10,100):
    for j in range(80,100):
        # img.itemset((i,j),0)
        img[i,j] = 255
cv2.imshow("after",img)
cv2.waitKey()
cv2.destroyAllWindows()