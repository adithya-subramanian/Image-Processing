import cv2
import numpy as np
from statistics import mode
def checkz(added,z):
    for m in range(len(added)):
        if added[m] == z:
            return False
    
    return True


img = cv2.imread('changed.jpg')   #loading image
edges = cv2.Canny(img,0.6*np.median(img),1.33*np.median(img)) #finding edges
im, contours1 ,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #finding contours i.e. the locations where text is present
kernel1 = np.uint8([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]]) # applying dilation continosuly until two subsequent iterations doesn't give same number of contours
dilation1 = cv2.dilate(edges,kernel1,iterations = 1)
prev1 = -1
while prev1 != len(contours1) :
	prev1 = len(contours1)
	dilation1 = cv2.dilate(dilation1,kernel1,iterations = 1)
	im, contours1 ,hierarchy = cv2.findContours(dilation1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

x1 = []  #starting x posn
y1 = []  #staring y posn
h1 = []  #height of box
w1 = []  #weight of box

for i in range(len(contours1)):
	x1.append([])
	y1.append([])
	h1.append([])
	w1.append([])

for i in range(len(contours1)):
	x1[i],y1[i],w1[i],h1[i] = cv2.boundingRect(contours1[i]) #creating bouding boxes 

nx1 = []
ny1 = []
nh1 = []
nw1 = []
inner1 = []
#removing boxes generated due to noise most of these noisy boxes lie inside a larger box so we are finding the boxes that lie inside a box
for i in range(len(x1)):
    for j in range(i+1,len(x1)):
        if ((x1[j] > x1[i]) or (x1[j] == x1[i])) and ((y1[j] > y1[i]) or (y1[j] == y1[i])) and ((y1[j]+h1[j] < y1[i]+h1[i]) or (y1[j]+h1[j] == y1[i]+h1[i])) and ((x1[j]+w1[j] < x1[i]+w1[i]) or (x1[j]+w1[j] == x1[i]+w1[i])) and checkz(inner1,j) :
            inner1.append(j)

for i in range(len(x1)):
    if checkz(inner1,i) :
        nx1.append(x1[i])
        ny1.append(y1[i])
        nh1.append(h1[i])
        nw1.append(w1[i])

##here after we finding the distinct location of y in sorted order
distinct_yposn = sorted(list(set(nx1))) 
cluster = []
locs = []
lists = []
ehs = []
hmax = []

#Storing the width of every element in different clusters 
for i in range(len(distinct_yposn)):
	for j in range(len(ny1)):
		if distinct_yposn[i] == nx1[j] :
			ehs.append(nw1[j])

#finding the maximum of all the width's for each cluster
for i in range(len(ehs)):
	hmax.append(np.max(ehs[i]))

#If the position of the cluster is
k = -1
for i in range(len(distinct_yposn)):
	if checkz(lists,i):
		k=k+1
		cluster.append([])
		cluster[k].append(i)
		lists.append(i)
        for j in range(len(i,distinct_yposn)):
		    #if (distinct_yposn[i] - 5*maxd < distinct_yposn[j] and distinct_yposn[i] + 5*maxd > distinct_yposn[j]) or distinct_yposn[i] - 5*maxd == distinct_yposn[j]  or distinct_yposn[i] + 5*maxd == distinct_yposn[j] and checkz(lists,j):
		    if (distinct_yposn[i] + hmax[i]/2 > distinct_yposn[j] and distinct_yposn[i] - hmax[i]/2 < distinct_yposn[j]) or distinct_yposn[i] - hmax[i]/2 == distinct_yposn[j]  or distinct_yposn[i] + hmax[i]/2 == distinct_yposn[j] and checkz(lists,j):
		        cluster[k].append(j)
		        lists.append(j)

#print len(cluster)

for i in range(len(cluster)):
	locs.append([])

for i in range(len(cluster)):
	for j in range(len(cluster[i])):
		for k in range(len(ny1)):
		    if distinct_yposn[cluster[i][j]] == nx1[k] :
		    	locs[i].append(k)

#print locs
for i in range(len(locs)):
	for j in range(len(locs[i])):
		cv2.rectangle(img,(nx1[locs[i][j]],ny1[locs[i][j]]),(nx1[locs[i][j]]+nw1[locs[i][j]],ny1[locs[i][j]]+nh1[locs[i][j]]),(0,255,0),2)
	cv2.imwrite('sad/kjl%d.jpg'%i,img)
	img = cv2.imread('changed.jpg')
