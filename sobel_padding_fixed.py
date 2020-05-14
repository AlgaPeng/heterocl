from PIL import Image
import heterocl as hcl
import numpy as np
import math
import imageio

hcl.init(init_dtype=hcl.Fixed(15,5))
path = "home.jpg"
img = Image.open(path)
width,height = img.size

A = hcl.placeholder((height+2,width+2,3), "A")
Gx = hcl.placeholder((3,3),"Gx")
Gy = hcl.placeholder((3,3),"Gy")

def sobel(A,Gx,Gy):

   B = hcl.compute((height+2,width+2), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B")	
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
   D = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*Gx[r,c],axis=[r,c]), B[x,y]), "Gx")
   t = hcl.reduce_axis(0, 3)
   g = hcl.reduce_axis(0, 3)
   E = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Gy[t,g],axis=[t,g]), B[x,y]), "Gy")
   
   return  hcl.compute((height,width), lambda x,y:(hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y]))/4328*255)

s = hcl.create_schedule([A,Gx,Gy],sobel)
f = hcl.build(s)

npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)
npA = np.array(img)
hcl_A = hcl.asarray(npA)

npF = np.zeros((height,width))
hcl_F = hcl.asarray(npF)
f(hcl_A, hcl_Gx,hcl_Gy, hcl_F)
npF = hcl_F.asnumpy()

newimg = np.zeros((height,width,3))
for x in range(0, height):
	for y in range(0, width):
		for z in range(0,3):
			newimg[x,y,z]=npF[x,y]

imageio.imsave("home_sobel_fixed.jpg",newimg)
