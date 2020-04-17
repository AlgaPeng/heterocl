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

#initialize placeholders
A = hcl.placeholder((height,width,3), "A", dtype=hcl.Float())
Gx = hcl.placeholder((3,3),"Gx",dtype=hcl.Float())
Gy = hcl.placeholder((3,3),"Gy",dtype=hcl.Float())
npA = np.asarray(img)
npGx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
npGy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
hcl_A = hcl.asarray(npA)
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)

#output
npF = np.zeros((height,width))
hcl_F = hcl.asarray(npF)


#=======================================sobel_x============================================
def sobel_x(A,Gx):

   B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B",dtype=hcl.Float())	
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
   return hcl.compute((height,width), lambda x,y: hcl.select(hcl.and_(x>0, x<(height -1), y>0, y<(width-1)), hcl.sum(B[x+r,y+c]*Gx[r,c], axis=[r,c]), B[x, y]),"X",dtype=hcl.Float())
   
sx = hcl.create_schedule([A,Gx],sobel_x)
fx = hcl.build(sx)

npX = np.zeros((height,width))
hcl_X = hcl.asarray(npX)

fx(hcl_A, hcl_Gx, hcl_X)
npX = hcl_X.asnumpy()

#=======================================sobel_y============================================

def sobel_y(A,Gy):   
    B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B",dtype=hcl.Float())	
    t = hcl.reduce_axis(0,3)
    g = hcl.reduce_axis(0,3)

    return  hcl.compute((height,width), lambda x,y:  hcl.select(hcl.and_(x>0, x<(height -1), y>0, y<(width-1)), hcl.sum(B[x+t,y+g]*Gy[t,g], axis=[t,g]), B[x, y]),"Y",dtype=hcl.Float())

sy = hcl.create_schedule([A,Gy],sobel_y)
fy = hcl.build(sy)

npY = np.zeros((height,width))
hcl_Y = hcl.asarray(npY)

fy(hcl_A, hcl_Gy, hcl_Y)
npY = hcl_Y.asnumpy()

#======================================================================================

G = np.hypot(npX, npY)
G = G/G.max()*255
theta = np.arctan2(npY,npX)

#output image
newimg = np.zeros((height,width,3))
for x in range(0, height):
	for y in range(0, width):
		for z in range(0,3):
			newimg[x,y,z]=theta[x,y]

newimg = newimg.astype(np.uint8)
imageio.imsave("Will_sobel.jpg",newimg)
