from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'));sys.path.insert(0,str(ROOT/'tmp/stabilization-v2-deps'))
import cv2,numpy as np
from scipy.optimize import least_squares
from analyze_cv import CONFIG
OUT=Path(__file__).resolve().parent/'v2';cv2.setNumThreads(4);M=np.load(OUT/'board-game-content-matrices-v5.npy');size=np.array([1300.,1040.]);D=np.diag([1299.,1039.,1.]);frames=np.arange(112,147);n=len(frames);sift=cv2.SIFT_create(5000,contrastThreshold=.009,edgeThreshold=18);cap=cv2.VideoCapture(str(CONFIG['board-game']['src']));features=[];mask=np.zeros((1040,1300),np.uint8);mask[25:-25,25:-25]=255;tic=time.time()
for k in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,int(k));ok,im=cap.read();assert ok;g=cv2.warpPerspective(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),D@M[k],(1300,1040));kp,d=sift.detectAndCompute(g,mask);features.append((np.float32([p.pt for p in kp]),d))
cap.release();I=[];J=[];A=[];B=[];matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=80));edges=0
for i in range(n):
 for delta in [1,2,3,5,8]:
  j=i+delta
  if j>=n:continue
  p,d=features[i];q,e=features[j]
  if d is None or e is None or min(len(d),len(e))<8:continue
  matches=[a for a,b in matcher.knnMatch(d,e,k=2) if a.distance<.74*b.distance]
  if len(matches)<10:continue
  a=p[[m.queryIdx for m in matches]];b=q[[m.trainIdx for m in matches]];h,inl=cv2.findHomography(a,b,cv2.USAC_MAGSAC,1.5,maxIters=3000,confidence=.999)
  if h is None:continue
  use=inl.ravel().astype(bool);a=a[use];b=b[use]
  if len(a)<10 or cv2.contourArea(cv2.convexHull(a))/np.prod(size)<.12:continue
  # Limit each region so the moving brush cannot dominate the plane.
  cells={};keep=[]
  for ii,p0 in enumerate(a):
   cell=tuple((p0/size*5).astype(int))
   if cells.get(cell,0)<6:keep.append(ii);cells[cell]=cells.get(cell,0)+1
  a=a[keep];b=b[keep];I.extend([i]*len(a));J.extend([j]*len(a));A.extend(a/size);B.extend(b/size);edges+=1
I=np.array(I);J=np.array(J);A=np.c_[A,np.ones(len(A))];B=np.c_[B,np.ones(len(B))];ident=np.array([1,0,0,0,1,0,0,0.]);fixed=set([n-1]);free=[i for i in range(n) if i not in fixed];x0=np.tile(ident,(len(free),1)).ravel()
def matrices(x):
 p=np.tile(ident,(n,1));p[free]=x.reshape(-1,8);return np.c_[p,np.ones(n)].reshape(n,3,3)
def project(P,H):
 z=np.einsum('nij,nj->ni',H,P);return z[:,:2]/z[:,2:3]
def fun(x):
 h=matrices(x);a=project(A,h[I]);b=project(B,h[J]);return np.r_[((a-b)*size).ravel(),(x-x0)*50]
print('EARLY GRAPH',edges,len(I),'observations',round(time.time()-tic,1),flush=True)
# Only 272 parameters in this short, difficult low-texture transition.
from scipy.sparse import lil_matrix
index={k:i for i,k in enumerate(free)};sp=lil_matrix((2*len(I)+len(x0),len(x0)))
for t,(i,j) in enumerate(zip(I,J)):
 for k in [i,j]:
  if k in index:sp[2*t:2*t+2,index[k]*8:index[k]*8+8]=1
for k in range(len(x0)):sp[2*len(I)+k,k]=1
r=least_squares(fun,x0,jac_sparsity=sp.tocsr(),loss='soft_l1',f_scale=.7,x_scale='jac',max_nfev=40,ftol=1e-5,verbose=1);h=matrices(r.x);M[frames]=h@M[frames];np.save(OUT/'board-game-content-matrices-v6.npy',M);np.save(OUT/'board-game-dense-matrices.npy',M);(OUT/'board-game-early-graph-v6.json').write_text(json.dumps({'frames':frames.tolist(),'matches':len(I),'edges':edges,'seconds':time.time()-tic,'success':bool(r.success)},indent=2));print('EARLY DONE',round(time.time()-tic,1),flush=True)
