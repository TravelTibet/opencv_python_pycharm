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
# img = cv2.imread("./pic/lena.bmp")
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

# import cv2
# img=cv2.imread("./pic/lena.bmp",0)
# #测试读取、修改单个像素值
# print("读取像素点 img.item(3,2)=",img.item(3,2))
# # img.itemset((3,2),255)
# img[3,2] = 255
# print("修改后像素点 img.item(3,2)=",img.item(3,2))
# #测试修改一个区域的像素值
# cv2.imshow("before",img)
# for i in range(10,100):
#     for j in range(80,100):
#         # img.itemset((i,j),0)
#         img[i,j] = 255
# cv2.imshow("after",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 提取 lena 脸部信息
# import cv2
# a=cv2.imread("./pic/lena.bmp",cv2.IMREAD_UNCHANGED)
# face=a[220:400,250:350]
# cv2.imshow("original",a)
# cv2.imshow("face",face)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 对lena脸部打码
# import cv2
# import numpy as np
#
# a = cv2.imread("./pic/lena.bmp", cv2.IMREAD_UNCHANGED)
# cv2.imshow("original", a)
# face = np.random.randint(0, 256, [180, 100,3])
# a[220:400, 250:350] = face
# cv2.imshow("result", a)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# lena = cv2.imread("./pic/lena.bmp",cv2.IMREAD_UNCHANGED)
# dollar = cv2.imread("./pic/dollar.bmp",cv2.IMREAD_UNCHANGED)
# cv2.imshow("lena",lena)
# cv2.imshow("dollar",dollar)
# face = lena[220:400,250:350]
# dollar[160:340,200:300] = face
# cv2.imshow("result",dollar)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# lena = cv2.imread("./pic/lena512.bmp",cv2.IMREAD_UNCHANGED)
# dollar = cv2.imread("./pic/dollar.bmp",cv2.IMREAD_UNCHANGED)
# cv2.imshow("lena",lena)
# cv2.imshow("dollar",dollar)
# print(dollar)
# face=lena[220:400,250:350]
# dollar[160:340,200:300]=face
# cv2.imshow("result",dollar)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
#
# lena = cv2.imread("./pic/lenacolor.png")
# cv2.imshow("lena", lena)
# b = lena[:, :, 0]
# g = lena[:, :, 1]
# r = lena[:, :, 2]
# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)
# lena[:, :, 0] = 0
# cv2.imshow("lena b 0", lena)
# lena[:, :, 1] = 0
# cv2.imshow("lena b 0 g 0", lena)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# lena = cv2.imread("./pic/lenacolor.png")
# cv2.imshow("lena1", lena)
# b = lena[:, :, 0]
# g = lena[:, :, 1]
# r = lena[:, :, 2]
# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)
# lena[:, :, 0] = 0
# cv2.imshow("lenab0", lena)
# lena[:, :, 1] = 0
# cv2.imshow("lenab0g0", lena)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
#
# lena = cv2.imread("pic/lena.bmp")
# b, g, r = cv2.split(lena)
# cv2.imshow("B",b)
# cv2.imshow("G",g)
# cv2.imshow("R",r)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# lena=cv2.imread("pic/lena.bmp")
# b,g,r=cv2.split(lena)
# bgr=cv2.merge([b,g,r])
# rgb=cv2.merge([r,g,b])
# cv2.imshow("lena",lena)
# cv2.imshow("bgr",bgr)
# cv2.imshow("rgb",rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# gray=cv2.imread("./pic/lena.tif",0)
# color=cv2.imread("./pic/lena.bmp")
# print("图像 gray 属性：")
# print("gray.shape=",gray.shape)
# print("gray.size=",gray.size)
# print("gray.dtype=",gray.dtype)
# print("图像 color 属性：")
# print("color.shape=",color.shape)
# print("color.size=",color.size)
# print("color.dtype=",color.dtype)


# import cv2
# import numpy as np
# a=cv2.imread("pic/lena.tif", 0)
# b=np.zeros(a.shape,dtype=np.uint8)
# b[100:400,200:400]=255
# b[100:500,100:200]=255
# c=cv2.bitwise_and(a,b)
# cv2.imshow("a",a)
# cv2.imshow("b",b)
# cv2.imshow("c",c)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# a = cv2.imread("./pic/lena.bmp", 1) # 3 通道
# b = np.zeros(a.shape, dtype=np.uint8)
# b[100:400, 200:400] = 255
# b[100:500, 100:200] = 255
# c = cv2.bitwise_and(a, b)
# print("a.shape = ", a.shape)
# print("b.shape = ", b.shape)
# cv2.imshow("a", a)
# cv2.imshow("b", b)
# cv2.imshow("c", c)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# lena = cv2.imread("./pic/lena.tif", 0)  # 灰度图
# cv2.imshow("lena", lena)
# r, c = lena.shape
# x = np.zeros((r, c, 8), dtype=np.uint8)
# for i in range(8):
#     x[:, :, i] = 2 ** i
# r = np.zeros((r, c, 8), dtype=np.uint8)
# for i in range(8):
#     r[:, :, i] = cv2.bitwise_and(lena, x[:, :, i])
#     mask = (r[:, :, i] > 0)
#     r[mask] = 255
#     cv2.imshow(str(i), r[:, :, i])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# lena = cv2.imread("./pic/lena.tif", 0)
# r, c = lena.shape
# key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
# # key = cv2.imread("./pic/lena.bmp",0)
# encryption = cv2.bitwise_xor(lena,key)
# decryption = cv2.bitwise_xor(encryption,key)
# cv2.imshow("lena",lena)
# cv2.imshow("key",key)
# cv2.imshow("encryption",encryption)
# cv2.imshow("decryption",decryption)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# lena = cv2.imread("./pic/lena.tif", 0)
# watermark = cv2.imread("./pic/watermark.bmp", 0)
# w = watermark[:, :] > 0  # w 是存放bool值的矩阵
# watermark[w] = 1  # 二值化处理
# r, c = lena.shape
# t254 = np.ones((r, c), dtype=np.uint8) * 254    # 对最后一位 置0
# lenaH7 = cv2.bitwise_and(lena,t254)
# e = cv2.bitwise_or(lenaH7,watermark)
# t1 = np.ones((r,c),dtype = np.uint8)
# wm=cv2.bitwise_and(e,t1)
# print(wm)
# w = wm[:,:] > 0
# wm[w] = 255
# cv2.imshow("lena",lena)
# cv2.imshow("watermark",watermark * 255)
# cv2.imshow("e",e)
# cv2.imshow("wm",wm)
# cv2.waitKey()
# cv2.destroyAllWindows()


# chapter 04 start

# import cv2
# import numpy as np
#
# img = np.random.randint(0, 256, size=[2, 4], dtype=np.uint8)
# rst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# print("img=\n", img)
# print("rst=\n", rst)


# import cv2
#
# lena = cv2.imread("./pic/lenacolor.png")
# gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
# rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# # ==========打印 shape============
# print("lena.shape=", lena.shape)
# print("gray.shape=", gray.shape)
# print("rgb.shape=", rgb.shape)
# # ==========显示效果============
# cv2.imshow("lena", lena)
# cv2.imshow("gray", gray)
# cv2.imshow("rgb", rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# lena = cv2.imread("./pic/lenacolor.png")
# rgb = cv2.cvtColor(lena,cv2.COLOR_BGR2RGB)
# cv2.imshow("lena",lena)
# cv2.imshow("rgb",rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# imgBlue = np.zeros([1, 1, 3], dtype=np.uint8)
# imgBlue[0, 0, 0] = 255
# Blue = imgBlue
# BlueHSV = cv2.cvtColor(Blue, cv2.COLOR_BGR2HSV)
# print("Blue =\n", Blue)
# print("BlueHSV =\n", BlueHSV)
# imgGreen = np.zeros([1, 1, 3], dtype=np.uint8)
# imgGreen[0, 0, 1] = 255
# Green = imgGreen
# GreenHSV = cv2.cvtColor(Green, cv2.COLOR_BGR2HSV)
# print("Green=\n", Green)
# print("GreenHSV=\n", GreenHSV)
# # =========测试一下 OpenCV 中红色的 HSV 模式值=============
# imgRed = np.zeros([1, 1, 3], dtype=np.uint8)
# imgRed[0, 0, 2] = 255
# Red = imgRed
# RedHSV = cv2.cvtColor(Red, cv2.COLOR_BGR2HSV)
# print("Red=\n", Red)
# print("RedHSV=\n", RedHSV)


# chapter 6 start

# import cv2
# import numpy as np
#
# img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
# t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# print("img = \n", img)
# print("t = ", t)
# print("res = \n", rst)


# import cv2
#
# img = cv2.imread("./pic/lena.tif")
# t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("img", img)
# cv2.imshow("rst", rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# img = cv2.imread("./pic/computer.jpg", 0)
# t1, thd = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print("t1 = ", t1)
# # t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# athdMean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
# athdGAUS = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
# cv2.imshow("img", img)
# cv2.imshow("thd", thd)
# cv2.imshow("athdMene", athdMean)
# cv2.imshow("athdGAUS", athdGAUS)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# img = cv2.imread("./pic/tiffany.bmp", 0)
# # img = cv2.imread("./pic/airfield2.bmp", 0)
# t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print("t2 = ", t2)
# cv2.imshow("img", img)
# cv2.imshow("thd", thd)
# cv2.imshow("otsu", otsu)
# cv2.waitKey()
# cv2.destroyAllWindows()


# chapter 6 ending


# chapter 8 start

# import cv2
# import numpy as np
#
# o = cv2.imread("./pic/erode.bmp", cv2.IMREAD_UNCHANGED)  # 返回原始图像。alpha通道不会被忽略，如果有的话。
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv2.erode(o, kernel)
# cv2.imshow("orriginal", o)
# cv2.imshow("erosion", erosion)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread("./pic/dilation.bmp", cv2.IMREAD_UNCHANGED)
# kernel = np.ones((5, 5), np.uint8)
# dilation = cv2.dilate(o, kernel, iterations=9)
# cv2.imshow("original", o)
# cv2.imshow("dilation", dilation)
# cv2.waitKey()
# cv2.destroyAllWindows()

# chapter 8 ending
# chapter 9,10 pass

# chapter 11 start


# import cv2
#
# o = cv2.imread("./pic/lena.bmp", cv2.IMREAD_GRAYSCALE)
# r1 = cv2.pyrDown(o)
# r2 = cv2.pyrDown(r1)
# r3 = cv2.pyrDown(r2)
# print("o.shape=", o.shape)
# print("r1.shape=", r1.shape)
# print("r2.shape=", r2.shape)
# print("r3.shape=", r3.shape)
# cv2.imshow("original", o)
# cv2.imshow("r1", r1)
# cv2.imshow("r2", r2)
# cv2.imshow("r3", r3)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# o = cv2.imread("./pic/lenas.bmp")
# r1 = cv2.pyrUp(o)
# r2 = cv2.pyrUp(r1)
# r3 = cv2.pyrUp(r2)
# print("o.shape=", o.shape)
# print("r1.shape=", r1.shape)
# print("r2.shape=", r2.shape)
# print("r3.shape=", r3.shape)
# cv2.imshow("original", o)
# cv2.imshow("r1", r1)
# cv2.imshow("r2", r2)
# cv2.imshow("r3", r3)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# o = cv2.imread("./pic/lena.bmp")
# down = cv2.pyrDown(o)
# up = cv2.pyrUp(down)
# diff = up - o  # 构造 diff 图像，查看 up 与 o 的区别
# print("o.shape=", o.shape)
# print("up.shape=", up.shape)
# cv2.imshow("original", o)
# cv2.imshow("up", up)
# cv2.imshow("difference", diff)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# o = cv2.imread("./pic/lena.bmp")
# up = cv2.pyrUp(o)
# down = cv2.pyrDown(up)
# diff = down - o  # 构造 diff 图像，查看 down 与 o 的区别
# diff1 = o - down  # 构造 diff 图像，查看 down 与 o 的区别
# print("o.shape=", o.shape)
# print("down.shape=", down.shape)
# cv2.imshow("original", o)
# cv2.imshow("down", down)
# cv2.imshow("difference", diff)
# cv2.imshow("difference1", diff1)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# O = cv2.imread("./pic/lena.bmp")
# G0 = O
# G1 = cv2.pyrDown(G0)
# G2 = cv2.pyrDown(G1)
# G3 = cv2.pyrDown(G2)
# L0 = G0 - cv2.pyrUp(G1)
# L1 = G1 - cv2.pyrUp(G2)
# L2 = G2 - cv2.pyrUp(G3)
# print("L0.shape = ", L0.shape)
# print("L1.shape = ", L1.shape)
# print("L2.shape = ", L2.shape)
# cv2.imshow("L0", L0)
# cv2.imshow("L1", L1)
# cv2.imshow("L2", L2)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# O = cv2.imread("./pic/lena.bmp")
# G0 = O
# G1 = cv2.pyrDown(G0)
# L0 = O - cv2.pyrUp(G1)
# RO = L0 + cv2.pyrUp(G1)
# print("O.shape = ", O.shape)
# print("RO.shape = ", RO.shape)
# result = RO - O
# result = abs(result) # 避免 负负为正
# print("原始图像O与恢复图像RO之差的绝对值和：", np.sum(result))

# chapter 11 ending


# chapter 12 start


# import cv2
#
# o = cv2.imread('./pic/contours.bmp')
# cv2.imshow("original", o)
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(image=binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
# o = cv2.drawContours(o, contours, -1, (0, 0, 255), 5)
# cv2.imshow("result", o)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread("./pic/contours.bmp")
# cv2.imshow("original", o)
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# n = len(contours)
# contoursImg = []
# for i in range(n):
#     temp = np.zeros(o.shape, np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255, 255, 255), 5)
#     cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread("./pic/loc3.jpg")
# cv2.imshow("original", o)
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# # cv2.imshow("binary",binary)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# mask = np.zeros(o.shape, np.uint8)
# mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
# cv2.imshow("mask", mask)
# loc = cv2.bitwise_and(o, mask)
# cv2.imshow("location", loc)
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread('./pic/moments.bmp')
# cv2.imshow("original", o)
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# n = len(contours)
# contoursImg = []
# for i in range(n):
#     temp = np.zeros(o.shape, np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255,0,0), 3)
#     cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
# print("观察各个轮廓的矩（moments）:")
# for i in range(n):
#     print("轮廓" + str(i) + "的矩:\n", cv2.moments(contours[i]))
# print("观察各个轮廓的面积:")
# for i in range(n):
#     print("轮廓" + str(i) + "的面积:%d" % cv2.moments(contours[i])['m00'])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread('./pic/contours.bmp')
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow("original", o)
# n = len(contours)
# contoursImg = []
# for i in range(n):
#     print("contours[" + str(i) + "]面积=", cv2.contourArea(contours[i]))
#     temp = np.zeros(o.shape, np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i] = cv2.drawContours(contoursImg[i],
#                                       contours,
#                                       i,
#                                       (255, 255, 255),
#                                       3)
#     if cv2.contourArea(contours[i]) > 15000:
#         cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# o = cv2.imread("./pic/contours0.bmp")
# cv2.imshow("original", o)
# gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# n = len(contours)
# cntLen = []
# for i in range(n):
#     cntLen.append(cv2.arcLength(contours[i], True))
#     print("第" + str(i) + "个轮廓的长度：{}".format(cntLen[i]))
# cntLenSum = np.sum(cntLen)
# cntLenAvr = cntLenSum / n
# print("轮廓的总长度为：{}".format(cntLenSum))
# print("轮廓的平均长度为：{}".format(cntLenAvr))
# contoursImg = []
# for i in range(n):
#     temp = np.zeros(o.shape, np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255, 255, 255), 3)
#     if cv2.arcLength(contours[i], True) > cntLenAvr:
#         cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
#
# o1 = cv2.imread('./pic/cs1.bmp')
# gray = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
# HuM1 = cv2.HuMoments(cv2.moments(gray)).flatten()
# print("cv2.moments(gray)=\n", cv2.moments(gray))
# print("\nHuM1=\n", HuM1)
# print("\ncv2.moments(gray)['nu20']+cv2.moments(gray)['nu02']=%f+%f=%f\n"
#       % (cv2.moments(gray)['nu20'], cv2.moments(gray)['nu02'],
#          cv2.moments(gray)['nu20'] + cv2.moments(gray)['nu02']))
# print("HuM1[0]=", HuM1[0])
# print("\nHu[0]-(nu02+nu20)=",
#       HuM1[0] - (cv2.moments(gray)['nu20'] + cv2.moments(gray)['nu02']))

# chapter 12.1 ending


