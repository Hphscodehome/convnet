import torch
#二维卷积无padding
def corr2d(Data,Kernel):
    Y=torch.zeros((Data.shape[0]-Kernel.shape[0]+1,Data.shape[1]-Kernel.shape[1]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(Data[i:i+Kernel.shape[0],j:j+Kernel.shape[1]]*Kernel).sum()
    return Y
#二维卷积有padding
#def corr2d_pad(Data=(a,b),Kernel=(c,d),Padding=(e,f),Stride=(g,h)):
def corr2d_pad(Data,Kernel,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1])),Data,torch.zeros((Padding[0],Data.shape[1]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1])),Data,torch.zeros((Data.shape[0],Padding[1]))),axis=1)
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0]+1)//(Stride[0]),(Data.shape[1]-Kernel.shape[1]+1)//(Stride[1])))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=corr2d(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1]],Kernel)
    return Y   
#三维卷积无padding
def corr3d(Data,Kernel):
    Y=torch.zeros((Data.shape[0]-Kernel.shape[0]+1,Data.shape[1]-Kernel.shape[1]+1,Data.shape[2]-Kernel.shape[2]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                Y[i,j,k]=(Data[i:i+Kernel.shape[0],j:j+Kernel.shape[1],k:k+Kernel.shape[2]]*Kernel).sum()
    return Y
#三维卷积有padding
def corr3d_pad(Data,Kernel,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1],Data.shape[2])),Data,torch.zeros((Padding[0],Data.shape[1],Data.shape[2]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1],Data.shape[2])),Data,torch.zeros((Data.shape[0],Padding[1],Data.shape[2]))),axis=1)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Data.shape[1],Padding[2])),Data,torch.zeros((Data.shape[0],Data.shape[1],Padding[2]))),axis=2)
    Y=torch.zeros(((Data.shape[0]-Kernel.shape[0]+1)//(Stride[0]),(Data.shape[1]-Kernel.shape[1]+1)//(Stride[1]),(Data.shape[2]-Kernel.shape[2]+1)//(Stride[2])))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                Y[i,j,k]=corr3d(Data[i*Stride[0]:i*Stride[0]+Kernel.shape[0],j*Stride[1]:j*Stride[1]+Kernel.shape[1],k*Stride[2]:k*Stride[2]+Kernel.shape[2]],Kernel)
    return Y
# 测试案例
#corr2d(torch.tensor([[1,2,3],[4,5,6],[5,6,7]]),torch.tensor([[1,2],[2,3]]))
#corr2d_pad(torch.tensor([[1,2,3],[4,5,6],[5,6,7]]),torch.tensor([[1,2],[2,3]]),(1,1),(1,1))
#corr3d(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=torch.tensor([[[1,2],[1,1]],[[1,-1],[-1,2]]]))
#corr3d_pad(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=torch.tensor([[[1,2],[1,1]],[[1,-1],[-1,2]]]),Padding=(0,0,0),Stride=(1,1,1))
####################################################
#池化
#二维池化无padding
#def pool2d(Data,Mode='max,'avg'):
def pool2d(Data,Kernel,mode):
    Y=torch.zeros((Data.shape[0]-Kernel[0]+1,Data.shape[1]-Kernel[1]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=Data[i:i+Kernel[0],j:j+Kernel[1]].max()
            elif mode=='mean':
                Y[i,j]=Data[i:i+Kernel[0],j:j+Kernel[1]].mean()
    return Y
#二维池化有padding
def   pool2d_pad(Data,Kernel,mode,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1])),Data,torch.zeros((Padding[0],Data.shape[1]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1])),Data,torch.zeros((Data.shape[0],Padding[1]))),axis=1)
    Y=torch.zeros(((Data.shape[0]-Kernel[0]+1)//(Stride[0]),(Data.shape[1]-Kernel[1]+1)//(Stride[1])))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].max()
            elif mode=='mean':
                Y[i,j]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1]].mean()
    return Y  
#三维池化无padding
def pool3d(Data,Kernel,mode):
    Y=torch.zeros((Data.shape[0]-Kernel[0]+1,Data.shape[1]-Kernel[1]+1,Data.shape[2]-Kernel[2]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if mode=='max':
                    Y[i,j,k]=Data[i:i+Kernel[0],j:j+Kernel[1],k:k+Kernel[2]].max()
                elif mode=='mean':
                    Y[i,j,k]=Data[i:i+Kernel[0],j:j+Kernel[1],k:k+Kernel[2]].mean()
    return Y
#三维池化有padding
def pool3d_pad(Data,Kernel,mode,Padding,Stride):
    Data=torch.concatenate((torch.zeros((Padding[0],Data.shape[1],Data.shape[2])),Data,torch.zeros((Padding[0],Data.shape[1],Data.shape[2]))),axis=0)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Padding[1],Data.shape[2])),Data,torch.zeros((Data.shape[0],Padding[1],Data.shape[2]))),axis=1)
    Data=torch.concatenate((torch.zeros((Data.shape[0],Data.shape[1],Padding[2])),Data,torch.zeros((Data.shape[0],Data.shape[1],Padding[2]))),axis=2)
    Y=torch.zeros(((Data.shape[0]-Kernel[0]+1)//(Stride[0]),(Data.shape[1]-Kernel[1]+1)//(Stride[1]),(Data.shape[2]-Kernel[2]+1)//(Stride[2])))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if mode=='max':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].max()
                elif mode=='mean':
                    Y[i,j,k]=Data[i*Stride[0]:i*Stride[0]+Kernel[0],j*Stride[1]:j*Stride[1]+Kernel[1],k*Stride[2]:k*Stride[2]+Kernel[2]].mean()
    return Y
# 测试案例
#pool2d(torch.tensor([[1.,2,3],[4,5,6],[5,6,7.]]),(2,3),'mean')
#pool2d_pad(torch.tensor([[1.,2,3],[4,5,6],[5,6,7]]),(2,2),'mean',(1,1),(1,1))
#pool3d(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=(2,2,2),mode='max')
#pool3d_pad(Data=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]]),Kernel=(2,2,2),mode='max',Padding=(0,0,0),Stride=(1,1,1))