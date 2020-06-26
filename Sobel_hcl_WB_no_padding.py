import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio

#================================================================================================================================================
#initialization
#================================================================================================================================================
path = "home.jpg"                                               # Your image path 
hcl.init(init_dtype=hcl.Float())
#hcl.init(init_dtype = hcl.Fixed(30, 16))
img = Image.open(path)
width, height = img.size

#================================================================================================================================================
#main function
#================================================================================================================================================

A = hcl.placeholder((height,width,3), "A")
B = hcl.placeholder((height, width), "B")  #input placeholder2
Gx = hcl.placeholder((3,3),"Gx")
Gy = hcl.placeholder((3,3),"Gy")

def sobel(A,B,Gx,Gy):	
   def img_mutate(x,y):
       B[x][y] = A[x][y][0]+A[x][y][1]+A[x][y][2]
   hcl.mutate(B.shape, lambda x,y: img_mutate(x,y))	
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
  # D = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*Gx[r,c],axis=[r,c]), B[x,y]), "xx")
   D = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c]), "xx")

   t = hcl.reduce_axis(0, 3)
   g = hcl.reduce_axis(0, 3)
  # E = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Gy[t,g],axis=[t,g]), B[x,y]), "yy")
   E = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g]), "yy")

   return  hcl.compute((height-2,width-2), lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867,"Fimg")

s = hcl.create_schedule([A,B,Gx,Gy],sobel)
WBX = s.reuse_at(B, s[sobel.xx], sobel.xx.axis[1], "WBX")
WBY = s.reuse_at(B, s[sobel.yy], sobel.yy.axis[1], "WBY")
f = hcl.build(s)

npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)
npA = np.array(img)
hcl_A = hcl.asarray(npA)
hcl_B = hcl.asarray(np.zeros((height,width)))
npF = np.zeros((height-2,width-2))
hcl_F = hcl.asarray(npF)
f(hcl_A, hcl_B, hcl_Gx, hcl_Gy, hcl_F)
npF = hcl_F.asnumpy()

#define array for image
newimg = np.zeros((height-2, width-2, 3))

#assign (length, length, length) to each pixel
for x in range (0, height-2):
        for y in range (0, width-2):
                for z in range (0, 3):
                        newimg[x,y,z]=npF[x,y]
imageio.imsave("home_sobel_WB.jpg",newimg)                        
                        
