import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
from tkinter import StringVar
import re
from tkinter import filedialog
import cv2
from PIL import Image


def resize_image(resize):
    max_size = (310, 310)
    resize.thumbnail(max_size)
    return resize

def create_page():
    global main
    main = tk.Frame(root, bg='#ffffff',bd=1)
    
    main.pack(side='right', fill='both', expand=True)
    # 创建水平分隔条，将窗口分为两部分
    
    global canvas
    canvas = tk.Canvas(root, bg='#666666',bd=1)
    
    canvas.configure(width=40)
    canvas.configure(width=40 + 20)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    # 创建侧边栏
    global sidebar
    sidebar = tk.Frame(canvas, bg='#ffffff',bd=1,relief=tk.RAISED)
    canvas.create_window((0, 0), window=sidebar, anchor="nw")
    
    # 在侧边栏中添加按钮（示例）
    #for i in range(20):
    #    button = tk.Button(sidebar, text="按钮{}".format(i+1))
    #    button.pack(pady=5)
    
    # 设置滚动条
    def set_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    # 监听窗口大小变化
    canvas.bind("<Configure>", set_scroll_region)

    # 在Canvas上添加事件处理程序
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_button_press(event):
        global prev_x, prev_y
        prev_x = event.x
        prev_y = event.y

    def on_button_motion(event):
        delta_x = event.x - prev_x
        delta_y = event.y - prev_y
        canvas.yview_scroll(-1*delta_y, "units")
        prev_x = event.x
        prev_y = event.y

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_button_motion)
    canvas.bind("<MouseWheel>", on_mousewheel)

    scrollbar.pack(side="left", fill="y")
    canvas.pack(side="left", fill="both")
    


def show_image():
    #图像框
    global r_image_show
    r_image_show = tk.Frame(raw_image, bg='#ffffff',width=310, height=310,bd=1,relief=tk.RAISED)
    r_image_show.pack(fill="both",expand=True)
    #显示图片
    global photo
    photo = tk.Label(r_image_show, image = img_png1)
    photo.pack(pady=5,padx=5,fill="both",expand=True)

    #图像框
    global n_image_show
    n_image_show = tk.Frame(new_image, bg='#ffffff', width=310, height=310,bd=1,relief=tk.RAISED)
    n_image_show.pack(fill="both",expand=True)
    #显示图片
    global label_img1
    label_img1 = tk.Label(n_image_show, image = img_png1)
    label_img1.pack(pady=5,padx=5,fill="both",expand=True)


def create_subarea():
    #左分区
    global raw_image_page
    raw_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image_page.pack(side='left', fill='both', expand=True)
    #图
    global raw_image
    raw_image = tk.Frame(raw_image_page, bg='#ffffff', width=310, height=310, bd=1, relief=tk.RAISED)
    raw_image.pack(side=tk.TOP, fill='both', expand=True)
    raw_image.pack_propagate(False)
    #组件
    global raw_module
    raw_module= tk.Frame(raw_image_page, bg='#ffffff', width=310, height=170, bd=1, relief=tk.RAISED)
    raw_module.pack(side=tk.TOP, fill='both', expand=True)
    raw_module.pack_propagate(False)



    #右分区
    global new_image_page
    new_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    new_image_page.pack(side='right', fill='both', expand=True)
    #图
    global new_image
    new_image = tk.Frame(new_image_page, bg='#ffffff', width=310, height=310, bd=1, relief=tk.RAISED)
    new_image.pack(side=tk.TOP, fill='both', expand=True)
    new_image.pack_propagate(False)
    #组件
    global new_module
    new_module= tk.Frame(new_image_page, bg='#ffffff', width=310, height=170, bd=1, relief=tk.RAISED)
    new_module.pack(side=tk.TOP, fill='both', expand=True)
    new_module.pack_propagate(False)


def loade_image(s):
    img_open1 = s

    img_open1 = resize_image(img_open1)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open1.size  # 获取调整后的图像大小
    
    resized_image = img_open1.resize((width, height))
    
    global img_png1

    img_png1 = ImageTk.PhotoImage(resized_image)



def histeq_image():
    img_open = save_img # 打开图片文件

    img_open = np.uint8(img_open)
    
    

    
    array_img = np.array(img_open)  # PIL.Image对象转为 Numpy 数组
    gray_array_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY) # 转为灰度图
    image1 = cv2.equalizeHist(gray_array_img) # 直方图均衡化
    
    # 计算直方图
    hist, _ = np.histogram(gray_array_img.ravel(), 256, [0,256])
    
    # 绘制直方图
    plt.figure()
    plt.hist(gray_array_img.ravel(), 256, [0, 256])
    plt.title("Histogram")
    
    # 将Matplotlib绘图转为PIL.Image对象
    fig = plt.gcf()
    fig.canvas.draw()
    hist_img = Image.frombytes('RGB', fig.canvas.get_width_height(),
                               fig.canvas.tostring_rgb())
    
    # 显示弹窗
    top = tk.Toplevel()
    img_label = tk.Label(top)
    img_label.pack()
    
    img_png = ImageTk.PhotoImage(hist_img)
    img_label.config(image=img_png)
    img_label.image = img_png  # 保存引用，防止被垃圾回收

















    

def func1(text):
    global save_img
    img_open = s #PIL

    img_open = resize_image(img_open)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open.size  # 获取调整前的图像大小
    
    
    array_img=np.array(img_open)#化为数组
    ratio=int(float(text))#获取参数
    
    imagel = np.zeros((int(array_img.shape[0]/ratio),int(array_img.shape[1]/ratio),array_img.shape[2]),dtype='int32')#采样后的大小
    for i in range (imagel.shape[0]):
        for j in range(imagel.shape[1]):
            for k in range(imagel.shape[2]):
                delta = array_img[i*ratio:(i+1)*ratio,j*ratio:(j+1)*ratio,k]#获取采样图像块
                #imagel[i,j,k]=np.mean(delta)#计算均值
                imagel[i, j, k] = np.max(delta)#最大值采样
    pil_img=Image.fromarray(np.uint8(imagel))#将一个Numpy数组转换为图像

    save_img=pil_img
    
    global img_png
    global label_img1



    pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img = pil_img.resize((width, height))
    pil_img=pil_img.resize((270, 210))#设置图片大小


    
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)


        

def func3():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
    
    pil_img=Image.fromarray(np.uint8(image1))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  

def func4(value):
    global save_img
    img_open = s #PIL

    array_img=np.array(img_open)
    value = input_box.get() #获取输入框的数据,阈值
    value_as_int = int(value)#整数化
    
    # 二值化
    new_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
    height, width = new_img.shape[0:2]

    # 设置阈值
    thresh = value_as_int

    # 遍历每一个像素点
    for row in range(height):
        for col in range(width):
            # 获取到灰度值
            gray = new_img[row, col]
            # 如果灰度值高于阈值 就等于255最大值
            if gray > thresh:
                new_img[row, col] = 255
            # 如果小于阈值，就直接改为0
            elif gray < thresh:
                new_img[row, col] = 0

    
    pil_img=Image.fromarray(np.uint8(new_img))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  
  
def func5left():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.rotate(array_img, cv2.ROTATE_90_CLOCKWISE)
    
    pil_img=Image.fromarray(np.uint8(image1))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  
def func5right():
    global save_img
    img_open = s #PIL

   

    
    array_img=np.array(img_open)

    
    
    image1 = cv2.rotate(array_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    pil_img=Image.fromarray(np.uint8(image1))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
    
#直方图
def func6():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    gray_array_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
    image1 =  cv2.equalizeHist(gray_array_img)
  
    pil_img=Image.fromarray(np.uint8(image1))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)



    
def func7():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)


  
    # 转换为灰度图像
    gray_image = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 混合图像
    blended_image = cv2.bitwise_and(array_img, array_img, mask=dilated_edges)



  
    pil_img=Image.fromarray(np.uint8(blended_image))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)








#镜像
def func8():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    mirror_img = cv2.flip(array_img, 1)
    
  
    pil_img=Image.fromarray(np.uint8(mirror_img))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)





def func9():
    global save_img
    img_open = s #PIL
    image1=np.array(img_open)

    h, w, c = image1.shape  # h=240  w=320
    src_list = [(61, 70), (151, 217), (269, 143), (160, 29)]
    for i, pt in enumerate(src_list):
        cv2.circle(image1, pt, 5, (0, 0, 255), -1)
        cv2.putText(image1,str(i+1),(pt[0]+5,pt[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


   
	
	
    pts1 = np.float32(src_list)

    pts2 = np.float32([[0, 0], [0, w - 2], [h - 2, w - 2], [h - 2, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image1, matrix, (h, w))


    
   
    
    pil_img=Image.fromarray(np.uint8(result))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  
        
def func10():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)





    image_float = image1.astype(np.float32)
    
    # 对数变换
    transformed = cv2.log(1 + image_float)
    
    # 将像素值缩放到 [0, 255]
    transformed = (transformed / np.max(transformed)) * 255
    
    # 将图像转换为 8 位无符号整型
    transformed = transformed.astype(np.uint8)


    
    pil_img=Image.fromarray(np.uint8(transformed))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  



def func11():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)



    gamma = 0.5#幂次值

    image_float = image1.astype(np.float32)
    
    #幂次变换
    transformed = cv2.pow(image_float, gamma)
    
    # 将像素值缩放到 [0, 255]
    transformed = (transformed / np.max(transformed)) * 255
    
    # 将图像转换为 8 位无符号整型
    transformed = transformed.astype(np.uint8)


    
    pil_img=Image.fromarray(np.uint8(transformed))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  




def func12():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)

    # 设置线性变换的参数
    alpha = 1.0
    beta = 50

    #线性变换
    transformed = cv2.convertScaleAbs(image1, alpha=alpha, beta=beta)


    
    pil_img=Image.fromarray(np.uint8(transformed))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)






def func13():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)


    dft = cv2.dft(np.float32(image1), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # 将低频部分移动到图像中心
    dft_shift = np.fft.fftshift(dft)
    
    # 计算幅度谱（频谱强度）
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    
    pil_img=Image.fromarray(np.uint8(magnitude_spectrum))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  

def func14():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    #image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)


    kernel_size = (3, 3)  # 结构元素的大小
    iterations = 1       # 迭代次数



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行腐蚀操作
    eroded = cv2.erode(array_img, kernel, iterations=iterations)
    
    
    pil_img=Image.fromarray(np.uint8(eroded))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  

def func15():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    #image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
    
    # 设置膨胀操作的参数
    kernel_size = (3, 3)  # 结构元素的大小
    iterations = 1       # 迭代次数



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行膨胀操作
    dilated = cv2.dilate(array_img, kernel, iterations=iterations)
    
    pil_img=Image.fromarray(np.uint8(dilated))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
          


def func16():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    #image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)

    # 设置开运算的参数
    kernel_size = (3, 3)  # 结构元素的大小
    iterations = 1       # 迭代次数



     # 创建腐蚀操作的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行腐蚀操作
    opened = cv2.morphologyEx(array_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    
    pil_img=Image.fromarray(np.uint8(opened))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  



def func17():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    #image1 = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)

    # 设置闭运算的参数
    kernel_size = (3, 3)  # 结构元素的大小
    iterations = 1       # 迭代次数



    # 创建膨胀操作的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行膨胀操作
    dilated = cv2.dilate(array_img, kernel, iterations=iterations)
    
    # 执行腐蚀操作
    closed = cv2.erode(dilated, kernel, iterations=iterations)
    
    
    pil_img=Image.fromarray(np.uint8(closed))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  


def func18():
    global save_img
    
    array_img2=np.array(s2)
    array_img3=np.array(s3)
    
    image1 = array_img2

    
    image2 = array_img3

    
    # 确保两个图像的大小相等
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 进行图像加运算
    result = cv2.add(image1, image2)
    
    
    pil_img=Image.fromarray(np.uint8(result))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  



def func19():
    global save_img
    
    array_img2=np.array(s2)
    array_img3=np.array(s3)
    
    image1 = array_img2

    
    image2 = array_img3

    
    # 确保两个图像的大小相等
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 进行图像加运算
    result = cv2.subtract(image1, image2)
    
    
    pil_img=Image.fromarray(np.uint8(result))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  


def func20():
    global save_img
    
    array_img2=np.array(s2)
    array_img3=np.array(s3)
    
    image1 = array_img2

    
    image2 = array_img3

    
    # 确保两个图像的大小相等
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 进行图像加运算
    result = cv2.bitwise_and(image1, image2)
    
    
    pil_img=Image.fromarray(np.uint8(result))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)




def func21():
    global save_img
    
    array_img2=np.array(s2)
    array_img3=np.array(s3)
    
    image1 = array_img2

    
    image2 = array_img3

    
    # 确保两个图像的大小相等
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # 进行图像加运算
    result = cv2.bitwise_or(image1, image2)
    
    
    pil_img=Image.fromarray(np.uint8(result))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)



def func22():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
    blurred = cv2.GaussianBlur(array_img, (3, 3), 0)

    # Canny边缘检测
    edges = cv2.Canny(blurred, 30, 70)
    
    
    pil_img=Image.fromarray(np.uint8(edges))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  


def func23():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)
    
   
    img_gray = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)  
       
    # Canny边缘检测  
    img_Canny = cv2.Canny(img_gray, 0, 250, (3, 3))  
       
    result_HoughLinesP = cv2.HoughLinesP(img_Canny, 1, 1 * np.pi / 180, 10, minLineLength=1, maxLineGap=5)  
  
    # 画出检测的线段  
    for result in result_HoughLinesP:  
        for x1, y1, x2, y2 in result:  
            cv2.line(array_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  

    
    pil_img=Image.fromarray(np.uint8(array_img))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  




def func24():
    global save_img
    img_open = s #PIL
    array_img=np.array(img_open)

    image=array_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 霍夫圆变换检测硬币
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)

    coin_count = 0

    # 绘制检测到的硬币
    if circles is not None:
        detected_coins = np.round(circles[0, :]).astype("int")
        coin_count = len(detected_coins)
        for (x, y, r) in detected_coins:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    # 在图像上添加硬币计数文本
    cv2.putText(image, f"Coins: {coin_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  


        
    pil_img=Image.fromarray(np.uint8(image))#将一个Numpy数组转换为图像
    save_img=pil_img
    global img_png
    global label_img1

    
    #pil_img = resize_image(pil_img)  # 调用 resize_image() 函数调整图像大小
    #width, height = pil_img.size  # 获取调整后的图像大小
    
    #pil_img=pil_img.resize((270, 210))#设置图片大小
    img_png = ImageTk.PhotoImage(pil_img)#获取PhotoImage对象
    
    if label_img1:
        label_img1.config(image=img_png)
    else:
        label_img1 = tk.Label(n_image_show, image=img_png)
        label_img1.pack(fill="both",expand=True)
  





def save_image():
    global save_img
    if save_img:
        filepath = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png", filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg"), ("GIF", "*.gif")))
        if filepath:
            save_img.save(filepath)


def open_image():
    global img_label, img
    filepath = filedialog.askopenfilename(title="选择图片文件", filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.gif"), ("All Files", "*.*")))
    if filepath:
        img = Image.open(filepath)
        global s
        s=img


        img = resize_image(img)  # 调用 resize_image() 函数调整图像大小
        width, height = img.size  # 获取调整后的图像大小
    
        img = img.resize((width, height))
        
        
        img = ImageTk.PhotoImage(img)#转换成适合的图片对象
        photo.config(image=img)
        label_img1.config(image=img)
        
        #photo.image = img

#加减与或
def open_image2():

    global photo2
    global s2
    global img_png2
    filepath = filedialog.askopenfilename(title="选择图片文件", filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.gif"), ("All Files", "*.*")))
    if filepath:
        img = Image.open(filepath)
        
        img = resize_image(img)  # 调用 resize_image() 函数调整图像大小
        width, height = img.size  # 获取调整后的图像大小
    
        img = img.resize((width, height))
        
        s2=img
        img = ImageTk.PhotoImage(img)#转换成适合的图片对象
        img_png2=img
   
        photo2.config(image=img)
        
        
        #photo.image = img


def open_image3():

    global photo3
    global s3
    global img_png3
    filepath = filedialog.askopenfilename(title="选择图片文件", filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.gif"), ("All Files", "*.*")))
    if filepath:
        img = Image.open(filepath)
    

        img = resize_image(img)  # 调用 resize_image() 函数调整图像大小
        width, height = img.size  # 获取调整后的图像大小
    
        img = img.resize((width, height))
        s3=img
        
        
        img = ImageTk.PhotoImage(img)#转换成适合的图片对象
        img_png3=img
       
        photo3.config(image=img)
       
        
        #photo.image = img


def upload_button():
    button1 = tk.Button(raw_module, text="导入图片", command=open_image)
    button2 = tk.Button(raw_module, text="保存图片", command=save_image)
    button3 = tk.Button(raw_module, text="保存直方图")
    button1.pack(pady=20)
    button2.pack(pady=15)
    button3.pack(pady=15)


def control_button1():
    
    scale1 = tk.Scale(new_module, from_=5, to=50, orient=tk.HORIZONTAL, tickinterval=10, length=200, variable=v,command=func1)
    scale1.pack(pady=20)
    scale1.set(5)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)

def control_button3():

    input = tk.Frame(new_module,bg='#ffffff', bd=1, relief=tk.RAISED)
    input.pack(expand=True)
    
    input_hint=tk.Label(input,text="输入阈值")
    global input_box
    input_box=tk.Entry(input,relief="solid")
    value = input_box.get()
    input_button=tk.Button(input, text="确认",command=lambda: func4(value))
    
    input_hint.pack(side='left')
    input_box.pack(side='left')   
    input_button.pack(side='left')

    
    input_below=tk.Frame(new_module,bg='#ffffff', bd=1, relief=tk.RAISED)
    input_below.pack(expand=True)
    
    button4 = tk.Button(input_below, text="直方图", command=histeq_image)
    button4.pack()

def control_button4():
    
    yes_button = tk.Button(new_module, text="确认",command=func3)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)





def control_button5():

    input = tk.Frame(new_module, bd=1, relief=tk.RAISED)
    input.pack(expand=True,pady=30)
    
   
    button_left=tk.Button(input,text="左转",command=func5left)
    button_right=tk.Button(input, text="右转",command=func5right)
    
   
    button_left.pack(side='left',padx=10)   
    button_right.pack(side='left',padx=10)

    
    input_below=tk.Frame(new_module,bg='#ffffff', bd=1, relief=tk.RAISED)
    input_below.pack(expand=True)
    
    button4 = tk.Button(input_below, text="直方图", command=histeq_image)
    button4.pack()


def control_button6():
    
    yes_button = tk.Button(new_module, text="确认",command=func7)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)


def control_button7():
    
    yes_button = tk.Button(new_module, text="确认",command=func8)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)


def control_button8():
    
    yes_button = tk.Button(new_module, text="确认",command=func9)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)



def control_button9():
    
    yes_button = tk.Button(new_module, text="确认",command=func10)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)



def control_button10():
    
    yes_button = tk.Button(new_module, text="确认",command=func11)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)

def control_button11():
    
    yes_button = tk.Button(new_module, text="确认",command=func12)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)




def control_button12():
    
    yes_button = tk.Button(new_module, text="确认",command=func13)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)




def control_button13():
    
    yes_button = tk.Button(new_module, text="确认",command=func14)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)




def control_button14():
    
    yes_button = tk.Button(new_module, text="确认",command=func15)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)






def control_button15():
    
    yes_button = tk.Button(new_module, text="确认",command=func16)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)





def control_button16():
    
    yes_button = tk.Button(new_module, text="确认",command=func17)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)



def control_button17():
    #左分区
    
    raw_image_page = tk.Frame(main, bd=5, relief=tk.RAISED, highlightbackground='black')
    raw_image_page.pack(side='left', fill='both', expand=True)


    #新
  
    raw_image = tk.Frame(raw_image_page)
    raw_image.pack(side=tk.TOP, fill='both', expand=True)
    raw_image.pack_propagate(False)
    #旧
  
    raw_module= tk.Frame(raw_image_page)
    raw_module.pack(side=tk.TOP, fill='both', expand=True)
    raw_module.pack_propagate(False)



    img_open1 = s
    img_open1 = resize_image(img_open1)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open1.size  # 获取调整后的图像大小
    
    resized_image = img_open1.resize((width, height))
    
    global img_png2
    global img_png3

    img_png2 = ImageTk.PhotoImage(resized_image)
    img_png3 = ImageTk.PhotoImage(resized_image)
    
    


    #图像框
    
    r_image_show = tk.Frame(raw_image)
    r_image_show.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片

    global photo2
    photo2 = tk.Label(r_image_show, image = img_png2)
    photo2.pack(pady=5,padx=5,fill="both",expand=True)


    #button1 = tk.Button(raw_image, text="导入图片1",width=50, height=50, command=open_image2)
    #button1.pack(side=tk.TOP, fill="both",expand=True)
    

    #图像框
  
    r_image_show2 = tk.Frame(raw_module)
    r_image_show2.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片
    global photo3
   
    photo3 = tk.Label(r_image_show2, image = img_png3)
    photo3.pack(pady=5,padx=5,fill="both",expand=True)



    


    #button2 = tk.Button(raw_module, text="导入图2片",width=50, height=50, command=open_image3)
    #button2.pack(side=tk.TOP, fill="both",expand=True)



########################################################################################

    #右分区

    new_image_page = tk.Frame(main, bd=5, relief=tk.RAISED, highlightbackground='black')
    new_image_page.pack(side='right', fill='both', expand=True)
    #图

    new_image = tk.Frame(new_image_page, width=250, height=310)
    new_image.pack(side=tk.TOP, fill='both', expand=True)
    new_image.pack_propagate(False)
    #组件
  
    new_module= tk.Frame(new_image_page,width=250, height=170)
    new_module.pack(side=tk.TOP, fill='both', expand=True)
    new_module.pack_propagate(False)



    #图像框
  
    n_image_show = tk.Frame(new_image)
    n_image_show.pack(fill="both",expand=True)
    #显示图片
    global label_img1
    label_img1 = tk.Label(n_image_show, image = img_png1)
    label_img1.pack(pady=5,padx=5,fill="both",expand=True)


    
    yes_button = tk.Button(new_module, text="确认",command=func18)
    yes_button.pack(pady=10)

    button4 = tk.Button(new_module, text="保存图片")
    button4.pack(pady=10)

    button1 = tk.Button(new_module, text="导入图片1", command=open_image2)
    button1.pack(pady=10)
    


    button2 = tk.Button(new_module, text="导入图2片", command=open_image3)
    button2.pack(pady=10)




def control_button18():
    #左分区
    
    raw_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image_page.pack(side='left', fill='both', expand=True)


    #新
  
    raw_image = tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image.pack(side=tk.TOP, fill='both', expand=True)
    raw_image.pack_propagate(False)
    #旧
  
    raw_module= tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_module.pack(side=tk.TOP, fill='both', expand=True)
    raw_module.pack_propagate(False)



    img_open1 = s
    img_open1 = resize_image(img_open1)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open1.size  # 获取调整后的图像大小
    
    resized_image = img_open1.resize((width, height))
    
    global img_png2
    global img_png3

    img_png2 = ImageTk.PhotoImage(resized_image)
    img_png3 = ImageTk.PhotoImage(resized_image)
    
    


    #图像框
    
    r_image_show = tk.Frame(raw_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片

    global photo2
    photo2 = tk.Label(r_image_show, image = img_png2)
    photo2.pack(pady=5,padx=5,fill="both",expand=True)


    #button1 = tk.Button(raw_image, text="导入图片1",width=50, height=50, command=open_image2)
    #button1.pack(side=tk.TOP, fill="both",expand=True)
    

    #图像框
  
    r_image_show2 = tk.Frame(raw_module, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show2.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片
    global photo3
   
    photo3 = tk.Label(r_image_show2, image = img_png3)
    photo3.pack(pady=5,padx=5,fill="both",expand=True)



    


    #button2 = tk.Button(raw_module, text="导入图2片",width=50, height=50, command=open_image3)
    #button2.pack(side=tk.TOP, fill="both",expand=True)



########################################################################################

    #右分区

    new_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    new_image_page.pack(side='right', fill='both', expand=True)
    #图

    new_image = tk.Frame(new_image_page, bg='#ffffff' , width=250, height=310,bd=1, relief=tk.RAISED)
    new_image.pack(side=tk.TOP, fill='both', expand=True)
    new_image.pack_propagate(False)
    #组件
  
    new_module= tk.Frame(new_image_page, bg='#ffffff',width=250, height=170, bd=1, relief=tk.RAISED)
    new_module.pack(side=tk.TOP, fill='both', expand=True)
    new_module.pack_propagate(False)



    #图像框
  
    n_image_show = tk.Frame(new_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    n_image_show.pack(fill="both",expand=True)
    #显示图片
    global label_img1
    label_img1 = tk.Label(n_image_show, image = img_png1)
    label_img1.pack(pady=5,padx=5,fill="both",expand=True)


    
    yes_button = tk.Button(new_module, text="确认",command=func19)
    yes_button.pack(pady=10)

    button4 = tk.Button(new_module, text="保存图片")
    button4.pack(pady=10)

    button1 = tk.Button(new_module, text="导入图片1", command=open_image2)
    button1.pack(pady=10)
    


    button2 = tk.Button(new_module, text="导入图2片", command=open_image3)
    button2.pack(pady=10)




def control_button19():
    #左分区
    
    raw_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image_page.pack(side='left', fill='both', expand=True)


    #新
  
    raw_image = tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image.pack(side=tk.TOP, fill='both', expand=True)
    raw_image.pack_propagate(False)
    #旧
  
    raw_module= tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_module.pack(side=tk.TOP, fill='both', expand=True)
    raw_module.pack_propagate(False)



    img_open1 = s
    img_open1 = resize_image(img_open1)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open1.size  # 获取调整后的图像大小
    
    resized_image = img_open1.resize((width, height))
    
    global img_png2
    global img_png3

    img_png2 = ImageTk.PhotoImage(resized_image)
    img_png3 = ImageTk.PhotoImage(resized_image)
    
    


    #图像框
    
    r_image_show = tk.Frame(raw_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片

    global photo2
    photo2 = tk.Label(r_image_show, image = img_png2)
    photo2.pack(pady=5,padx=5,fill="both",expand=True)


    #button1 = tk.Button(raw_image, text="导入图片1",width=50, height=50, command=open_image2)
    #button1.pack(side=tk.TOP, fill="both",expand=True)
    

    #图像框
  
    r_image_show2 = tk.Frame(raw_module, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show2.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片
    global photo3
   
    photo3 = tk.Label(r_image_show2, image = img_png3)
    photo3.pack(pady=5,padx=5,fill="both",expand=True)



    


    #button2 = tk.Button(raw_module, text="导入图2片",width=50, height=50, command=open_image3)
    #button2.pack(side=tk.TOP, fill="both",expand=True)



########################################################################################

    #右分区

    new_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    new_image_page.pack(side='right', fill='both', expand=True)
    #图

    new_image = tk.Frame(new_image_page, bg='#ffffff' , width=250, height=310,bd=1, relief=tk.RAISED)
    new_image.pack(side=tk.TOP, fill='both', expand=True)
    new_image.pack_propagate(False)
    #组件
  
    new_module= tk.Frame(new_image_page, bg='#ffffff',width=250, height=170, bd=1, relief=tk.RAISED)
    new_module.pack(side=tk.TOP, fill='both', expand=True)
    new_module.pack_propagate(False)



    #图像框
  
    n_image_show = tk.Frame(new_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    n_image_show.pack(fill="both",expand=True)
    #显示图片
    global label_img1
    label_img1 = tk.Label(n_image_show, image = img_png1)
    label_img1.pack(pady=5,padx=5,fill="both",expand=True)


    
    yes_button = tk.Button(new_module, text="确认",command=func20)
    yes_button.pack(pady=10)

    button4 = tk.Button(new_module, text="保存图片")
    button4.pack(pady=10)

    button1 = tk.Button(new_module, text="导入图片1", command=open_image2)
    button1.pack(pady=10)
    


    button2 = tk.Button(new_module, text="导入图2片", command=open_image3)
    button2.pack(pady=10)







def control_button20():
    #左分区
    
    raw_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image_page.pack(side='left', fill='both', expand=True)


    #新
  
    raw_image = tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_image.pack(side=tk.TOP, fill='both', expand=True)
    raw_image.pack_propagate(False)
    #旧
  
    raw_module= tk.Frame(raw_image_page, bg='#ffffff', bd=1, relief=tk.RAISED)
    raw_module.pack(side=tk.TOP, fill='both', expand=True)
    raw_module.pack_propagate(False)



    img_open1 = s
    img_open1 = resize_image(img_open1)  # 调用 resize_image() 函数调整图像大小
    width, height = img_open1.size  # 获取调整后的图像大小
    
    resized_image = img_open1.resize((width, height))
    
    global img_png2
    global img_png3

    img_png2 = ImageTk.PhotoImage(resized_image)
    img_png3 = ImageTk.PhotoImage(resized_image)
    
    


    #图像框
    
    r_image_show = tk.Frame(raw_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片

    global photo2
    photo2 = tk.Label(r_image_show, image = img_png2)
    photo2.pack(pady=5,padx=5,fill="both",expand=True)


    #button1 = tk.Button(raw_image, text="导入图片1",width=50, height=50, command=open_image2)
    #button1.pack(side=tk.TOP, fill="both",expand=True)
    

    #图像框
  
    r_image_show2 = tk.Frame(raw_module, bg='#ffffff',bd=1,relief=tk.RAISED)
    r_image_show2.pack(side=tk.TOP, fill="both",expand=True)
    #显示图片
    global photo3
   
    photo3 = tk.Label(r_image_show2, image = img_png3)
    photo3.pack(pady=5,padx=5,fill="both",expand=True)



    


    #button2 = tk.Button(raw_module, text="导入图2片",width=50, height=50, command=open_image3)
    #button2.pack(side=tk.TOP, fill="both",expand=True)



########################################################################################

    #右分区

    new_image_page = tk.Frame(main, bg='#ffffff', bd=1, relief=tk.RAISED)
    new_image_page.pack(side='right', fill='both', expand=True)
    #图

    new_image = tk.Frame(new_image_page, bg='#ffffff' , width=250, height=310,bd=1, relief=tk.RAISED)
    new_image.pack(side=tk.TOP, fill='both', expand=True)
    new_image.pack_propagate(False)
    #组件
  
    new_module= tk.Frame(new_image_page, bg='#ffffff',width=250, height=170, bd=1, relief=tk.RAISED)
    new_module.pack(side=tk.TOP, fill='both', expand=True)
    new_module.pack_propagate(False)



    #图像框
  
    n_image_show = tk.Frame(new_image, bg='#ffffff',bd=1,relief=tk.RAISED)
    n_image_show.pack(fill="both",expand=True)
    #显示图片
    global label_img1
    label_img1 = tk.Label(n_image_show, image = img_png1)
    label_img1.pack(pady=5,padx=5,fill="both",expand=True)


    
    yes_button = tk.Button(new_module, text="确认",command=func21)
    yes_button.pack(pady=10)

    button4 = tk.Button(new_module, text="保存图片")
    button4.pack(pady=10)

    button1 = tk.Button(new_module, text="导入图片1", command=open_image2)
    button1.pack(pady=10)
    


    button2 = tk.Button(new_module, text="导入图2片", command=open_image3)
    button2.pack(pady=10)





def control_button21():
    
    yes_button = tk.Button(new_module, text="确认",command=func22)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)



def control_button22():
    
    yes_button = tk.Button(new_module, text="确认",command=func23)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)




def control_button23():
    
    yes_button = tk.Button(new_module, text="确认",command=func24)
    yes_button.pack(pady=30)

    button4 = tk.Button(new_module, text="直方图", command=histeq_image)
    button4.pack(pady=20)
    


   
def empty():
    for child in main.winfo_children(): #所有子组件
        child.destroy()  # 逐个删除子组件



#测试
'''
def fun1():
    empty()
    
    
def fun2():
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button()
'''

def fun1():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button1()

def fun2():
    empty()

def fun3():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button3()


def fun4():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button4()



def fun5():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button5()

def fun6():
    empty()
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button6()


    
def fun7():
    empty()
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button7()


def fun8():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button8()

    



def fun9():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button9()


def fun10():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button10()


def fun11():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button11()



def fun12():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button12()

def fun13():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button13()


def fun14():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button14()


def fun15():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button15()


def fun16():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button16()



def fun17():
    empty()
   
    control_button17()


def fun18():
    empty()
    control_button18()


def fun19():
    empty()
    control_button19()
      
def fun20():
    empty()
    control_button20()



def fun21():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button21()



def fun22():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button22()

def fun23():
    empty()
    create_subarea()
    loade_image(s)
    show_image()
    upload_button()
    control_button23()
    

def load_button():
    button1 = tk.Button(sidebar, text='量化',command=fun1, width=8, height=2)
    button1.pack(fill='x')

    button2 = tk.Button(sidebar, text='清空',command=fun2, width=8, height=2)
    button2.pack(fill='x')

    button3 = tk.Button(sidebar, text='二值',command=fun3, width=8, height=2)
    button3.pack(fill='x')

    button4 = tk.Button(sidebar, text='黑白',command=fun4, width=8, height=2)
    button4.pack(fill='x')

    button5 = tk.Button(sidebar, text='旋转',command=fun5, width=8, height=2)
    button5.pack(fill='x')

    button6 = tk.Button(sidebar, text='增强',command=fun6, width=8, height=2)
    button6.pack(fill='x')

    button7 = tk.Button(sidebar, text='镜像',command=fun7, width=8, height=2)
    button7.pack(fill='x')

    button8 = tk.Button(sidebar, text='透视',command=fun8, width=8, height=2)
    button8.pack(fill='x')

    button9 = tk.Button(sidebar, text='对数',command=fun9, width=8, height=2)
    button9.pack(fill='x')

    button10 = tk.Button(sidebar, text='幂次',command=fun10, width=8, height=2)
    button10.pack(fill='x')

    button11 = tk.Button(sidebar, text='线性',command=fun11, width=8, height=2)
    button11.pack(fill='x')

    button12 = tk.Button(sidebar, text='傅里叶',command=fun12, width=8, height=2)
    button12.pack(fill='x')

    button13 = tk.Button(sidebar, text='腐蚀',command=fun13, width=8, height=2)
    button13.pack(fill='x')

    button14 = tk.Button(sidebar, text='膨胀',command=fun14, width=8, height=2)
    button14.pack(fill='x')
    
    button15 = tk.Button(sidebar, text='开运算',command=fun15, width=8, height=2)
    button15.pack(fill='x')

    button16 = tk.Button(sidebar, text='闭运算',command=fun16, width=8, height=2)
    button16.pack(fill='x')

    button17 = tk.Button(sidebar, text='加',command=fun17, width=8, height=2)
    button17.pack(fill='x')

    button18 = tk.Button(sidebar, text='减',command=fun18, width=8, height=2)
    button18.pack(fill='x')

    button19 = tk.Button(sidebar, text='与',command=fun19, width=8, height=2)
    button19.pack(fill='x')

    button20 = tk.Button(sidebar, text='或',command=fun20, width=8, height=2)
    button20.pack(fill='x')

    button20 = tk.Button(sidebar, text='边缘',command=fun21, width=8, height=2)
    button20.pack(fill='x')


    button21 = tk.Button(sidebar, text='霍夫变换',command=fun22, width=8, height=2)
    button21.pack(fill='x')

    button22 = tk.Button(sidebar, text='检测和计数',command=fun23, width=8, height=2)
    button22.pack(fill='x')



img_png1=None

img_png2=None
img_png3=None
save_img=None

img = None # 用于存储导入的图片


root = tk.Tk()
v=StringVar()


root.geometry("900x576+300+100")#大小，（高度，宽度）+（x轴，y轴）
#root.resizable(False, False)#不可放大窗口

#a = Image.open("t2.png")
#s  = resize_image(a)  # 调用 resize_image() 函数调整图像大小
 
s = Image.open("t2.png")

s2=s
s3=s

save_img=s

create_page()
load_button()
create_subarea()
loade_image(s)
show_image()
upload_button()
control_button1()


root.mainloop()

