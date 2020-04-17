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

hcl_A = hcl.asarray(np.asarray(img))
hcl_Gx = hcl.asarray(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
hcl_Gy = hcl.asarray(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

#output
hcl_F = hcl.asarray(np.zeros((height,width)))

#=======================================sobel_x==============================================
def sobel_x(A,Gx):

   B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B",dtype=hcl.Float())	
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
   return hcl.compute((height,width), lambda x,y: hcl.select(hcl.and_(x>0, x<(height -1), y>0, y<(width-1)), hcl.sum(B[x+r,y+c]*Gx[r,c], axis=[r,c]), B[x,y]),"X",dtype=hcl.Float())
   
sx = hcl.create_schedule([A,Gx],sobel_x)
fx = hcl.build(sx)

hcl_X = hcl.asarray(np.zeros((height,width)))

fx(hcl_A, hcl_Gx, hcl_X)
npX = hcl_X.asnumpy()

#=======================================sobel_y===============================================

def sobel_y(A,Gy):   
    B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2],"B",dtype=hcl.Float())	
    t = hcl.reduce_axis(0,3)
    g = hcl.reduce_axis(0,3)

    return  hcl.compute((height,width), lambda x,y:  hcl.select(hcl.and_(x>0, x<(height -1), y>0, y<(width-1)), hcl.sum(B[x+t,y+g]*Gy[t,g], axis=[t,g]), B[x, y]),"Y",dtype=hcl.Float())

sy = hcl.create_schedule([A,Gy],sobel_y)
fy = hcl.build(sy)

hcl_Y = hcl.asarray(np.zeros((height,width)))

fy(hcl_A, hcl_Gy, hcl_Y)
npY = hcl_Y.asnumpy()

#============================================================================================

G = np.hypot(npX, npY)
G = G/G.max()*255
theta = np.arctan2(npY,npX)

#===================================Non_Max===================================================
#Edge Direction 

#placeholder for non_max

angle = hcl.placeholder((height,width),"angle")

for i in range(1,height-1):
    for j in range(1,width-1):
        d = theta[i][j]*180. / np.pi
        if(d<0):
            theta[i][j] = d+180
        

def non_max_suppression(angle):

    for i in range(1,height-1):
        for j in range(1,width-1):
                with hcl.if_():
                    pass
                q = 255
                r = 255
                
               #angle 0
                with hcl.if_ (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = theta[i, j+1]
                    r = theta[i, j-1]
                #angle 45
                with hcl.elif_ (22.5 <= angle[i,j] < 67.5):
                    q = theta[i+1, j-1]
                    r = theta[i-1, j+1]
                #angle 90
                with hcl.elif_ (67.5 <= angle[i,j] < 112.5):
                    q = theta[i+1, j]
                    r = theta[i-1, j]
                #angle 135
                with hcl.elif_ (112.5 <= angle[i,j] < 157.5):
                    q = theta[i-1, j-1]
                    r = theta[i+1, j+1]

                with hcl.if_ (theta[i,j] >= q) and (theta[i,j] >= r):
                    Z[i,j] = theta[i,j]

                with hcl.else_:
                    Z[i,j] = 0
        
    return Z

sd = hcl.create_schedule([angle],non_max_suppression)
fd = hcl.build(sd)

hcl_D = hcl.asarray(np.zeros((height,width)))
fx(theta, hcl_D)


#output image
newimg = np.zeros((height,width,3))
for x in range(0, height):
	for y in range(0, width):
		for z in range(0,3):
			newimg[x,y,z]=theta[x,y]

newimg = newimg.astype(np.uint8)
imageio.imsave("Will_sobel.jpg",newimg)
