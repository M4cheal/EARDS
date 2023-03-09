import torch
import torch.nn as nn
import torchvision.transforms as T

def double_conv(in_,out_,drop):
	conv = nn.Sequential(
		nn.Conv2d(in_,out_,kernel_size=3,padding=(1,1)),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_,out_,kernel_size=3,padding=(1,1)),
		nn.ReLU(inplace=True),
		nn.Dropout(drop)
		)
	return conv

def crop(tensor,target_tensor):
	target_shape = target_tensor.shape[2]
	return T.CenterCrop(target_shape)(tensor)

class UNet(nn.Module):
	def __init__(self,dropout=0.1):
		super(UNet,self).__init__()

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.enc_conv_1 = double_conv(1,64,dropout)
		self.enc_conv_2= double_conv(64,128,dropout)
		self.enc_conv_3 = double_conv(128,256,dropout)
		self.enc_conv_4 = double_conv(256,512,dropout)
		self.enc_conv_5 = double_conv(512,1024,dropout)

		self.up_trans_1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
		self.dec_conv_1 = double_conv(1024,512,dropout)
		self.up_trans_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
		self.dec_conv_2 = double_conv(512,256,dropout)
		self.up_trans_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
		self.dec_conv_3 = double_conv(256,128,dropout)
		self.up_trans_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
		self.dec_conv_4 = double_conv(128,64,dropout)

		self.out = nn.Conv2d(64,2,kernel_size=1)

	def forward(self, image):
		# Encoder
		x1 = self.enc_conv_1(image)
		x = self.pool(x1)
		x2 = self.enc_conv_2(x)
		x = self.pool(x2)
		x3 = self.enc_conv_3(x)
		x = self.pool(x3)
		x4 = self.enc_conv_4(x)
		x = self.pool(x4)
		x = self.enc_conv_5(x)

		#Decoder
		x = self.up_trans_1(x)
		x = self.dec_conv_1(torch.cat([x,crop(x4,x)],axis=1))
		x = self.up_trans_2(x)
		x = self.dec_conv_2(torch.cat([x,crop(x3,x)],axis=1))
		x = self.up_trans_3(x)
		x = self.dec_conv_3(torch.cat([x,crop(x2,x)],axis=1))
		x = self.up_trans_4(x)
		x = self.dec_conv_4(torch.cat([x,crop(x1,x)],axis=1))
		
		#out
		x = self.out(x)
		return x


# image = torch.rand((1,1,576,576))
# model = UNet()
# out = model(image)
# print(out.shape)




		 
