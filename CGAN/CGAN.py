from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#导入全连接网络层
from tensorflow.keras.layers import LeakyReLU#导入激活层
from tensorflow.keras.layers import BatchNormalization#导入BN层，即执行批量归一化操作
import numpy as np
from tensorflow.keras.layers import Reshape#导入Reshape层
from tensorflow.keras.layers import Input#导入Input层，指定输入维度
from tensorflow.keras.layers import Embedding#导入Embedding层
from tensorflow.keras.layers import Flatten#导入Flatten层
from tensorflow.keras.layers import multiply#导入multiply
from tensorflow.keras.models import Model#导入Model模型
from tensorflow.keras.layers import Dropout#导入Dropout层
from tensorflow.keras.datasets import mnist#导入mnist数据集
from tensorflow.keras.optimizers import Adam#导入Adam激活函数
import matplotlib.pyplot as plt

#定义CGAN类

class CGAN():
    def __init__(self):

        #写入输入维度
        self.img_rows=28 #图像的行是28个像素点
        self.img_cols=28 #图像的列是28个像素点
        self.img_channels=1 #图像的通道数是1，单通道的灰度图
        self.img_shape=(self.img_rows,self.img_cols,self.img_channels)#图像的尺寸

        self.num_classes=10#定义标签类别数
        self.latent_dim=100#定义输入生成器的噪声输入维度
        
        optimizer=Adam(0.0002,0.5)#将学习率设置维0.0002
        
        self.generator=self.build_generator() 
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])#使用二进制交叉熵损失函数

        self.discriminator.trainable=False
        noise=Input(shape=(100,))
        label=Input(shape=(1,))
        
        img=self.generator([noise,label])#利用噪声，通过生成器生成图像

        valid=self.discriminator([img,label])

        self.combined=Model([noise,label],valid)
        self.combined.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer)#构建完成联合模型
        

    def build_generator(self):#定义生成器，输入的是噪声，生成的是图像

        model=Sequential()

        #输入层
        model.add(Dense(256,input_dim=self.latent_dim))#256个神经元的全连接层，输入维度为100
        model.add(LeakyReLU(alpha=0.2))#激活层
        model.add(BatchNormalization(momentum=0.8))#动量设置为0.8
        #第二层
        model.add(Dense(512))#512个神经元的全连接层
        model.add(LeakyReLU(alpha=0.2))#激活层
        model.add(BatchNormalization(momentum=0.8))#动量设置为0.8
        #第三层
        model.add(Dense(1024))#1024个神经元的全连接层
        model.add(LeakyReLU(alpha=0.2))#激活层
        model.add(BatchNormalization(momentum=0.8))#动量设置为0.8
        #输出层
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))#指定输出图像大小（维度）及tanh激活函数
        model.add(Reshape(self.img_shape))#Reshape成图像尺寸

        model.summary()#记录参数情况

        noise=Input(shape=(self.latent_dim,))#生成器的输入维度为100
        label=Input(shape=(1,),dtype='int32')#定义标签维度为一维，标签类型是int型

        label_embedding=Flatten()(Embedding(self.num_classes,self.latent_dim)(label))#输入维度为10,输出维度为100,保证其与噪声具有同样的维度，功能是将10个种类（词向量种类）的label映射到100维（即维度映射操作）
        #将100维转换为（None,100），这里的None会随着batch而改变

        model_input=multiply([noise,label_embedding])#将噪声和embedding之后的label标签合并（同为100维），合并方法为对应位置相乘

        img=model(model_input)#生成图片

        return Model([noise,label],img)#输入按noise和label标签，合并则有内部完成
    
    
    def build_discriminator(self):#定义判别器

        model=Sequential()

        #输入层
        model.add(Dense(512,input_dim=np.prod(self.img_shape)))#np.prod将图像的长宽及通道数相乘，获得输入维度  784个输入神经元
        model.add(LeakyReLU(alpha=0.2))
        #第二层
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))#防止模型过拟合，提高泛化能力
        #第三层
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        #输出层
        model.add(Dense(1,activation='sigmoid'))#使用一个神经元

        model.summary()#查看参数相关内容

        img=Input(shape=self.img_shape)
        label=Input(shape=(1,),dtype='int32')
        
        #label与img的shape不同

        label_embedding=Flatten()(Embedding(self.num_classes,np.prod(self.img_shape))(label))#Embedding操作的输入为num_classes，输出为np.prod(self.img_shape)
        #label_embedding shape (None,784)
        flat_img=Flatten()(img)

        model_input=multiply([flat_img,label_embedding])#完成了对应元素相乘，shape(None,784)
        
        validity=model(model_input)#获取输出概率结果

        return Model([img,label],validity)#合并和维度操作是由模型内部完成的

    def train(self,epochs,batch_size=128,sample_interval=50):#训练判别器和生成器
        
        #获取数据集
        (X_train,Y_train,),(_,_)=mnist.load_data()

        #将获取到的图像转化为-1到1的区间内进行操作
        X_train=(X_train.astype(np.float32)-127.5)/127.5#设置训练集的类型为浮点型，灰度值范围是0-255
        X_train=np.expand_dims(X_train,axis=3)#扩展维度，在第3维进行扩展
        #将60000*28*28维度的图像扩展为60000*28*28*1（60000是第0维）
        
        Y_train=Y_train.reshape(-1,1)#-1自动计算第0维的维度空间数，也就是自动计算得出60000，将Y_train reshape成60000*1的标签

        #写入真实输出与虚假输出

        valid=np.ones((batch_size,1))
        fake=np.zeros((batch_size,1))

        for epoch in range(epochs):

            #优先训练判别器

            #从0-60000中随机获取batch_size个索引数，完成从60000张图片中随机挑选32张图片
            idx=np.random.randint(0,X_train.shape[0],batch_size)
            imgs,labels=X_train[idx],Y_train[idx]
            #完成了随机获取batch_size个图像以及对应的标签
            #imgs shape (batch_size,28,28,1)
            #labels shape (batch_size,1)

            noise=np.random.normal(0,1,(batch_size,self.latent_dim))#获取符合（0，1）正态分布的随机噪声shape(batch_size,100),送入生成器
            
            gen_imgs=self.generator.predict([noise,labels])#虚假图片和真实标签送入判别器，真实图片和真实标签训练判别器

            d_loss_real=self.discriminator.train_on_batch([imgs,labels],valid)#输入真实图片的loss
            d_loss_fake=self.discriminator.train_on_batch([gen_imgs,labels],fake)#输入虚假图片的loss
            d_loss=0.5*np.add(d_loss_real,d_loss_fake)
            #到这里完成了判别器的训练
            
            #训练生成器

            sampled_label=np.random.randint(0,10,batch_size).reshape(-1,1)#随机生成样本标签,-1自动计算batch_size的个数

            #固定判别器，训练生成器——在联合模型中
            g_loss=self.combined.train_on_batch([noise,sampled_label],valid)#生成器的loss

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"%(epoch,d_loss[0],100*d_loss[1],g_loss))
            #绘制进度图

            if epoch % sample_interval==0:#每50轮
                self.sample_images(epoch)#调用保存图像的函数，完成图像保存


    def sample_images(self,epoch):#定义保存图像的函数
        r,c=2,5 #输出两行五列 0-9每个标签下的一张图像（10张指定图像）
        noise=np.random.normal(0,1,(r*c,100))#一个batch是10 生成100维的噪声
        sampled_labels=np.arange(0,10).reshape(-1,1)#标签数是10个标签

        gen_imgs=self.generator.predict([noise,sampled_labels])#利用生成器生成图像

        #Rescale images 0-1
        gen_imgs=0.5*gen_imgs+0.5

        fig,axs=plt.subplots(r,c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].set_title("Digit: %d" %sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig("./images/images%d.png" % epoch)
        plt.close()

if __name__=='__main__':
    cgan=CGAN()
    cgan.train(epochs=20000,batch_size=32,sample_interval=200)#每200轮保存一下生成的10张图片（也就是已知标签的生成图像）



