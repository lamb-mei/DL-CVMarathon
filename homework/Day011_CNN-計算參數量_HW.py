# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 『本次練習內容』
# #### 運用Keras搭建簡單的Dense Layer與 Convolution2D Layer，使用相同Neurons數量，計算總參數量相差多少。
# 
# %% [markdown]
# ## 『本次練習目的』
#   #### 本次練習主要是要讓各位同學們了解CNN與FC層的參數使用量差異
#   #### Convolution2D有許多參數可以設置，之後章節會細談
#   #### 不熟Keras的學員們也可以藉此了解NN層的不同搭法

# %%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model

# 引用是相同的
# Convolution1D = Conv1D
# Convolution2D = Conv2D
# Convolution3D = Conv3D
# SeparableConvolution2D = SeparableConv2D
# Convolution2DTranspose = Conv2DTranspose
# Deconvolution2D = Deconv2D = Conv2DTranspose
# Deconvolution3D = Deconv3D = Conv3DTranspose


# %%
##輸入照片尺寸==28*28*1
##都用一層，288個神經元

##建造一個一層的CNN層
classifier=Sequential()


# keras.layers.convolutional.Conv2D()
# model.add(Conv2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
# input_shape 讓輸入的影像有一致的格式，程式中的 3 是指 R/B/G 三個通道
# activation function 設定為 ReLU （註：在[魔法陣系列] 王者誕生：AlexNet 之術式解析有介紹過 ReLU 的優點。）
# 用 max pooling，作用是降躁跟減少運算資源，設為 2 * 2 尺寸
# 一個 Convolution Operation 搭配 一個 Pooling

# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
# kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
# strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
# classifier.add(Convolution2D(64, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
##Kernel size 3*3，用32張，輸入大小28*28*1
# classifier.add(Convolution2D(32 , 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(Convolution2D(32 , 3, 3, input_shape = (28, 28, 1)))
'''32張Kernel，大小為3*3，輸入尺寸為28*28*1'''
##看看model結構
print(classifier.summary())
'''理解輸出Total params為何==320'''

#建造一個一層的FC層
classifier=Sequential()
##輸入為28*28*1攤平==784

# inputs = Input(shape=(28,28,1))  ＃矩陣錯誤的X
inputs = Input(shape=(784,))

'''輸入尺寸為28*28*1'''
##CNN中用了(3*3*1)*32個神經元，我們這邊也用相同神經元數量
x=Dense((3*3*1)*32)(inputs)
# x=Dense(32, activation='softmax')(inputs)
'''使用288個神經元'''
model = Model(inputs=inputs, outputs=x)
##看看model結構
print(model.summary())
'''理解輸出Total params為何==226080'''

##大家可以自己變換設定看看參數變化


# %%


