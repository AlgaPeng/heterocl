#heterocl Canny Filter
from PIL import Image
import heterocl as hcl
import numpy as np
import math
import imageio

hcl.init(init_dtype=hcl.Float())

#image path
path = "Will.jpg"
img = Image.open(path)
width,height = img.size
size = int(5)//2
sigma = 1
x,y = np.mgrid[-size:size+1, -size:size+1]
normal = 1 / (2.0 * np.pi * sigma**2)
g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

#===========================guassian===========================================

#guassian placeholders
A = hcl.placeholder((height,width), "A", dtype=hcl.Float())
G = hcl.placeholder((size,size),"G")

def guassian(A,G):
    h = hcl.reduce_axis(0,size)
    w = hcl.reduce_axis(0,size)
    return hcl.compute((height,width), lambda x,y: hcl.select(hcl.and_(x>(size-1), x<(height-size), y>(size-1),y<(width-size)), hcl.sum(A[x+h,y+w]*G[h,w], axis=[h,w]),A[x,y]),"F",dtype=hcl.Float())	
   
s = hcl.create_schedule([A,G],guassian)
f = hcl.build(s)

npA = np.array(img)
hcl_A = hcl.asarray(npA)
hcl_G = hcl.asarray(g)
npF = np.zeros((height,width))
hcl_F = hcl.asarray(npF)

f(hcl_A, hcl_G, hcl_F)
 
#============================sobel=============================================    
#sobel placeholders
Gx = hcl.placeholder((3,3),"Gx",dtype=hcl.Float())
Gy = hcl.placeholder((3,3),"Gy",dtype=hcl.Float())

def sobel(B,G):    
    r = hcl.reduce_axis(0,3)
    c = hcl.reduce_axis(0,3)
    return  hcl.compute((height,width), lambda x,y: hcl.select(hcl.and_(x>0, x<(height-1), y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*G[r,c], axis=[r,c]),B[x,y]),"D",dtype=hcl.Float())

s = hcl.create_schedule([A,Gx],sobel)
f = hcl.build(s)

npGx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
npGy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)
hcl_Sx = hcl.asarray(np.zeros((height,width)))
hcl_Sy = hcl.asarray(np.zeros((height,width)))

f(hcl_F,hcl_Gx,hcl_Sx)
f(hcl_F,hcl_Gy,hcl_Sy)

npSx = hcl_Sx.asnumpy()
npSy = hcl_Sy.asnumpy()
theta = np.arctan2(npSy, npSx)

#==============================================================================

#output image
newimg = np.zeros((height,width,3))
for x in range(0, height):
	for y in range(0, width):
		for z in range(0,3):
			newimg[x,y,z]=npS[x,y]

newimg = newimg.astype(np.uint8)
imageio.imsave("Will_sobel.jpg",newimg)
