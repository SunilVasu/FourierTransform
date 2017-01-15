#Question (d) Discuss why or why not the MSE may be non-zero
#Answer: The MSE is non zero in this solution because there are certain changes that has creeped in while
#the image is processed. for example if a image pixel calculation would vary after normalization form that 
#of the original pixel values. The mathematical operation on each pixel would cause alterations.
# Ideally the MSE is to be zero, but in real case due to losses while processing the image the MSE would #remian non zero.

#Note: As I was having problems with opncv in ubuntu 16.04 I have substituted the operation using opencv 
#with python functions likes numpy, Image etc. Due to this the image to be processed needs to be a very #small one as its would take more time to process the output.

import cv2
import numpy as np
import cmath
from PIL import Image
from matplotlib import pyplot as plt

def DFT(image):
	global M, N
	R = np.amax(image) #for logarithmic representation
	c= (255)/(cmath.log10(1+R))
	(M,N)=image.shape
	dft_result=[[0.0 for k in range(M)] for l in range(N)]
	pixels = image
	imageNew=Image.new("L",(M,N))
	pixelsNew=imageNew.load()
	for k in range(M):
		for l in range(N):
			sum=0.0
			for m in range(M):
				for n in range(N):
					#print pixels[m,n]
					e=cmath.exp(2*cmath.pi*-1j*(float(k*m)/M+float(l*n)/N))
					sum = sum + (c* cmath.log10(1+pixels[m,n]) * e)
					#print "Here",k,l
			dft_result[l][k]=sum
			pixelsNew[l,k]=int(sum.real)
	#q1=pixelsNew[:M/2,:N/2]
	#q2=pixelsNew[M/2:,:N/2]
	#q3=pixelsNew[0:M/2,N/2:]
	#q4=pixelsNew[M/2:,N/2:]
	
	#pixelsNew[:M/2,:N/2] =q1
	#pixelsNew[M/2:,:N/2] =q2
	#pixelsNew[0:M/2,N/2:]=q3
	#pixelsNew[M/2:,N/2:] =q4
	imageNew.save('2dDFT.jpg')
	imageNew.show()
	return dft_result


def IDFT(DFT2d):
	global M,N
	image = Image.new("L", (M, N))
	pixels=image.load()
	for m in range(M):
		for n in range(N):
			sum=0.0
			for k in range(M):
				for l in range(N):
					e=cmath.exp(2*cmath.pi*1j*(float(k*m)/M+float(l*n)/N))
					sum+=DFT2d[k][l]*e
			sum=(sum)/(N*M)
			pixels[m,n]=int(sum.real)
			
	return image

def MSE(image1,image2):
	(X,Y)=image1.shape
	pixels1=image1
	pixels2=image2.load()
	mse=0.0
	for m in range(X):
		for n in range(Y):
			mse+=(pixels1[m,n]-pixels2[m,n])*(pixels1[m,n]-pixels2[m,n])			
	return mse	


image = cv2.imread('small2.jpg',0)
cv2.imshow("Original Image",image)
array = np.asarray(image)
print(array)
rows=len(array)
array.shape
print(array.shape)


img= Image.open('small2.jpg')
img.show()

DFT2d=DFT(array)


IDFT2d=IDFT(DFT2d)
IDFT2d.save('2dIDFT.jpg')
IDFT2d.show()

mse=MSE(array,IDFT2d)
print "MSE=",mse


	
			










	
			










