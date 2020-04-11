#===================================import==================================================
from PIL import Image
import heterocl as hcl
import numpy as np
import math
import imageio
#==================================initialization============================================
hcl.init(init_dtype=hcl.Float())

#image path
path = "pic.jpg"
img = Image.open(path)
width,height = img.size

#initialize placeholders
A = hcl.placeholder((height,width,3), "A", dtype=hcl.Float())
Gx = hcl.placeholder((3,3),"Gx",dtype=hcl.Float())
Gy = hcl.placeholder((3,3),"Gy",dtype=hcl.Float())

#=======================================sobel_algo========================================================================
def sobel(A,Gx,Gy):

   B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B",dtype=hcl.Float())	
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
   D = hcl.compute((height-2,width-2), lambda x,y: hcl.sum(B[x+r,y+c]*Gx[r,c], axis=[r,c]),"D",dtype=hcl.Float())
   t = hcl.reduce_axis(0,3)
   g = hcl.reduce_axis(0,3)

   E = hcl.compute((height-2,width-2), lambda x,y: hcl.sum(B[x+t,y+g]*Gy[t,g], axis=[t,g]),"E",dtype=hcl.Float())
   return  hcl.compute((height-2,width-2), lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])/4328*255,dtype=hcl.Float())


#==========================================================================================================================

s = hcl.create_schedule([A,Gx,Gy],sobel)
f = hcl.build(s)

#filters
npA = np.array(img)
npGx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
npGy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
hcl_A = hcl.asarray(npA)
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)

#output
npF = np.zeros((height-2,width-2))
hcl_F = hcl.asarray(npF)

#call the function
f(hcl_A, hcl_Gx,hcl_Gy, hcl_F)
npF = hcl_F.asnumpy()

#output image
newimg = np.zeros((height-2,width-2,3))
for x in range(0, height-2):
	for y in range(0, width-2):
		for z in range(0,3):
			newimg[x,y,z]=npF[x,y]

newimg = newimg.astype(np.uint8)
imageio.imsave("pic_sobel.jpg",newimg)
