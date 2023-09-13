import torch
#二维卷积无padding
def corr2d(Data,Kernel,Stride):
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0])//(Stride[0])+1,(Data.shape[1]-Kernel.shape[1])//(Stride[1])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1]]*Kernel).sum()
    return Y
# 测试案例
#corr2d(torch.tensor([[1,2,3],[4,5,6],[5,6,7]]),torch.tensor([[1,2],[2,3]]),Stride=(1,1))
#二维卷积有padding
#def corr2d_pad(Data=(a,b),Kernel=(c,d),Padding=(e,f),Stride=(g,h)):
def corr2d_pad(Data,Kernel,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1])),Data,torch.zeros((Padding[0],Data.shape[1]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1])),Data,torch.zeros((Data.shape[0],Padding[1]))),axis=1)
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0])//(Stride[0])+1,(Data.shape[1]-Kernel.shape[1])//(Stride[1])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1]]*Kernel).sum()
    return Y
# 测试案例
#corr2d_pad(torch.tensor([[1,2,3],[4,5,6],[5,6,7]]),torch.tensor([[1,2],[2,3]]),(1,1),(1,1))
#三维卷积无padding
def corr3d(Data,Kernel,Stride):
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0])//(Stride[0])+1,(Data.shape[1]-Kernel.shape[1])//(Stride[1])+1,(Data.shape[2]-Kernel.shape[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                Y[i,j,k]=(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1],k*Stride[2]:k*Stride[2]+Kernel.shape[2]]*Kernel).sum()
    return Y
# 测试案例
#corr3d(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=torch.tensor([[[1,2],[1,1]],[[1,-1],[-1,2]]]),Stride=(1,1,1))
#三维卷积有padding
def corr3d_pad(Data,Kernel,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1],Data.shape[2])),Data,torch.zeros((Padding[0],Data.shape[1],Data.shape[2]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1],Data.shape[2])),Data,torch.zeros((Data.shape[0],Padding[1],Data.shape[2]))),axis=1)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Data.shape[1],Padding[2])),Data,torch.zeros((Data.shape[0],Data.shape[1],Padding[2]))),axis=2)
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0])//(Stride[0])+1,(Data.shape[1]-Kernel.shape[1])//(Stride[1])+1,(Data.shape[2]-Kernel.shape[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                Y[i,j,k]=(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1],k*Stride[2]:k*Stride[2]+Kernel.shape[2]]*Kernel).sum()
    return Y
# 测试案例
#corr3d_pad(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=torch.tensor([[[1,2],[1,1]],[[1,-1],[-1,2]]]),Padding=(0,0,0),Stride=(1,1,1))
####################################################
#池化
#二维池化无padding
#def pool2d(Data,Mode='max,'avg'):
def pool2d(Data,Kernel,mode,Stride):
    Y=torch.zeros(((Data.shape[0]-Kernel[0])//(Stride[0])+1,(Data.shape[1]-Kernel[1])//(Stride[1])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].max()
            elif mode=='mean':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].mean()
    return Y
# 测试案例
#pool2d(torch.tensor([[1.,2,3],[4,5,6],[5,6,7.]]),(2,3),'mean',Stride=(1,1))
#二维池化有padding
def pool2d_pad(Data,Kernel,mode,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1])),Data,torch.zeros((Padding[0],Data.shape[1]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1])),Data,torch.zeros((Data.shape[0],Padding[1]))),axis=1)
    Y=torch.zeros(((Data.shape[0]-Kernel[0])//(Stride[0])+1,(Data.shape[1]-Kernel[1])//(Stride[1])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].max()
            elif mode=='mean':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].mean()
    return Y
# 测试案例
#pool2d_pad(torch.tensor([[1.,2,3],[4,5,6],[5,6,7]]),(2,2),'mean',(1,1),(1,1))
#三维池化无padding
def pool3d(Data,Kernel,mode,Stride):
    Y=torch.zeros(((Data.shape[0]-Kernel[0])//(Stride[0])+1,(Data.shape[1]-Kernel[1])//(Stride[1])+1,(Data.shape[2]-Kernel[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if mode=='max':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].max()
                elif mode=='mean':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].mean()
    return Y
# 测试案例
#pool3d(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=(2,2,2),mode='max',Stride=(1,1,1))
#三维池化有padding
def pool3d_pad(Data,Kernel,mode,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1],Data.shape[2])),Data,torch.zeros((Padding[0],Data.shape[1],Data.shape[2]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1],Data.shape[2])),Data,torch.zeros((Data.shape[0],Padding[1],Data.shape[2]))),axis=1)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Data.shape[1],Padding[2])),Data,torch.zeros((Data.shape[0],Data.shape[1],Padding[2]))),axis=2)
    Y=torch.zeros(((Data.shape[0]-Kernel[0])//(Stride[0])+1,(Data.shape[1]-Kernel[1])//(Stride[1])+1,(Data.shape[2]-Kernel[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if mode=='max':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].max()
                elif mode=='mean':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].mean()
    return Y
# 测试案例
#pool3d_pad(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=(2,2,2),mode='max',Padding=(0,0,0),Stride=(1,1,1))
#二维测试
#Data=torch.randn(5,5)
#Kernel=torch.randn(3,3)
#Stride=(1,1)
#Padding=(1,1)
#print(Data,Kernel,Stride)
#print((Data[1:4,1:4]*Kernel).sum())
#Y=corr2d_pad(Data,Kernel,Stride,Padding)
#print(Y)
#Z=pool2d_pad(Y,Kernel=(3,3),mode='max',Padding=(1,1),Stride=(1,1))
#print(Z)
#三维测试
#Data=torch.randn(5,5,5)
#Kernel=torch.randn(3,3,3)
#Stride=(1,1,1)
#Padding=(1,1,1)
#print(Data,Kernel,Padding,Stride)
#print((Data[1:4,1:4,2:5]*Kernel).sum())
#Y=corr3d_pad(Data=Data,Kernel=Kernel,Padding=Padding,Stride=Stride)
#print(Y)
#Z=pool3d_pad(Y,Kernel=(3,3,3),mode='max',Padding=(1,1,1),Stride=(1,1,1))
#print(Z)
#定义卷积层操作无padding
#下面是二维样本特征时多样本多输出的卷积层
def convnet_2d(Data,Kernel,bias,Padding,Stride):
    Y=torch.zeros((Data.shape[0],Kernel.shape[0],(Data.shape[1]-Kernel.shape[1]+2*Padding[0])//(Stride[0])+1,(Data.shape[2]-Kernel.shape[2]+2*Padding[1])//(Stride[1])+1))
    #第几个样本
    for i in range(Y.shape[0]):
        #第几个输出内核
        for j in range(Y.shape[1]):
            Y[i,j,:,:]=corr2d_pad(Data[i,:,:].reshape(Data.shape[1:]),Kernel[j,:,:].reshape(Kernel.shape[1:]),Padding,Stride)
            Y[i,j,:,:]+=bias[j]        
    Y=1/(1+torch.exp(-Y))
    return Y
#测试
#Kernel=torch.randn((4,3,3),requires_grad=True)
#bias=torch.randn((Kernel.shape[0],1),requires_grad=True)
#convnet_2d(torch.randn(1,7,7),Kernel,bias,Padding=(1,1),Stride=(1,1))
#定义完整的二维样本卷积层
#def conv_2d(Data=(样本数,输入通道数,特征),Kernel=(输出通道数,输入通道数,二维卷积核),bias=(输出通道数,1),Padding=(二维扩充尺寸),Stride=(二维步进尺寸))->(样本数,输出通道数,卷积结果)
def convnet2d(Data,Kernel,bias,Padding,Stride):
    Y=torch.zeros((Data.shape[0],Kernel.shape[0],(Data.shape[2]-Kernel.shape[2]+2*Padding[0])//Stride[0]+1,(Data.shape[3]-Kernel.shape[3]+2*Padding[1])//Stride[1]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:,:]=corr3d_pad(Data[i,:,:,:].reshape(Data.shape[1:]),Kernel[j,:,:,:].reshape(Kernel.shape[1:]),(0,Padding[0],Padding[1]),(1,Stride[0],Stride[1])).reshape(Y.shape[2],Y.shape[3])
            Y[i,j,:,:]+=bias[j]
    Y=1/(1+torch.exp(-Y))
    return Y
#每个(输出通道,输入通道)组合都有一个偏置还是每个输出通道才有偏置，还是每个输入通道有偏置？还是每个卷积核都有一个偏置？
#前面的输出矩阵的尺寸容易理解，找几个例子推算一下就清楚了，
#下面是三维样本特征时多样本多输出的卷积层
def convnet_3d(Data,Kernel,bias,Padding,Stride):
    Y=torch.zeros((Data.shape[0],Kernel.shape[0],(Data.shape[1]-Kernel.shape[1]+2*Padding[0])//(Stride[0])+1,(Data.shape[2]-Kernel.shape[2]+2*Padding[1])//(Stride[1])+1,(Data.shape[3]-Kernel.shape[3]+2*Padding[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:,:,:]=corr3d_pad(Data[i,:,:,:].reshape(Data.shape[1:]),Kernel[j,:,:,:].reshape(Kernel.shape[1:]),Padding,Stride)
            Y[i,j,:,:,:]+=bias[j]
    Y=1/(1+torch.exp(-Y))
    return Y
#测试
#Kernel=torch.randn((4,3,3,3),requires_grad=True)
#bias=torch.randn((Kernel.shape[0],1),requires_grad=True)
#convnet_3d(torch.randn(1,7,7,7),Kernel,bias,Padding=(1,1,1),Stride=(1,1,1))
#下面是二维样本特征多样本多输出的汇聚层。
def poolnet_2d(Data,Kernel,mode,Padding,Stride):
    Y=torch.zeros((Data.shape[0],Data.shape[1],(Data.shape[2]-Kernel[0]+2*Padding[0])//(Stride[0])+1,(Data.shape[3]-Kernel[1]+2*Padding[1])//(Stride[1])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:,:]=pool2d_pad(Data[i,j,:,:].reshape(Data.shape[2:]),Kernel,mode,Padding,Stride)
    return Y
#测试
#Kernel=torch.randn((4,3,3),requires_grad=True)
#bias=torch.randn((Kernel.shape[0],1),requires_grad=True)
#Y=convnet_2d(torch.randn(1,7,7),Kernel,bias,Padding=(1,1),Stride=(1,1))
#print(Y)
#Z=poolnet_2d(Y,Kernel=(3,3),mode='max',Padding=(1,1),Stride=(1,1))
#print(Z)
#下面是三维样本特征多样本多输出的汇聚层
def poolnet_3d(Data,Kernel,mode,Padding,Stride):
    Y=torch.zeros((Data.shape[0],Data.shape[1],(Data.shape[2]-Kernel[0]+2*Padding[0])//(Stride[0])+1,(Data.shape[3]-Kernel[1]+2*Padding[1])//(Stride[1])+1,(Data.shape[4]-Kernel[2]+2*Padding[2])//(Stride[2])+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j,:,:,:]=pool3d_pad(Data[i,j,:,:,:].reshape(Data.shape[2:]),Kernel,mode,Padding,Stride)
    return Y
#测试
#Kernel=torch.randn((4,3,3,3),requires_grad=True)
#bias=torch.randn((Kernel.shape[0],1),requires_grad=True)
#Y=convnet_3d(torch.randn(1,7,7,7),Kernel,bias,Padding=(1,1,1),Stride=(1,1,1))
#print(Y)
#Z=poolnet_3d(Y,Kernel=(3,3,3),mode='max',Padding=(1,1,1),Stride=(1,1,1))
#print(Z)
if name==__main__:
    #def convnet_2d(Data,Kernel,bias,Padding,Stride)
    #def poolnet_2d(Data,Kernel,mode,Padding,Stride)
    #def convnet_3d(Data,Kernel,bias,Padding,Stride)
    #def poolnet_3d(Data,Kernel,mode,Padding,Stride)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(1)
    Data1=torch.randn((4,1,28,28))
    Kernel1=torch.randn((6,1,5,5),requires_grad=True)
    bias1=torch.randn((Kernel1.shape[0],1),requires_grad=True)
    Padding1=(2,2)
    Stride1=(1,1)
    C1=convnet2d(Data=Data1,Kernel=Kernel1,bias=bias1,Padding=Padding1,Stride=Stride1)
    print(C1.shape)
    P1=poolnet_2d(C1,Kernel=(2,2),mode='mean',Padding=(0,0),Stride=(2,2))
    print(P1.shape)
    Kernel2=torch.randn((16,6,5,5),requires_grad=True)
    bias2=torch.randn((Kernel2.shape[0],1),requires_grad=True)
    C2=convnet2d(P1,Kernel2,bias2,Padding=(0,0),Stride=(1,1))
    print(C2.shape)
    P2=poolnet_2d(C2,Kernel=(2,2),mode='mean',Padding=(0,0),Stride=(2,2))
    print(P2.shape)
    Y=torch.zeros((P2.shape[0],P2.shape[1]*P2.shape[2]*P2.shape[3]))
    for i in range(P2.shape[0]):
        Y[i,:]=P2[i,:,:,:].reshape(1,-1)
    weight1=torch.randn((Y.shape[1],120),requires_grad=True)
    Y1=1/(1+torch.exp(-torch.mm(Y,weight1)))
    weight2=torch.randn((Y1.shape[1],84),requires_grad=True)
    Y2=1/(1+torch.exp(-torch.mm(Y1,weight2)))
    weight3=torch.randn((Y2.shape[1],10),requires_grad=True)
    Y3=1/(1+torch.exp(-torch.mm(Y2,weight3)))
    print(Y3)