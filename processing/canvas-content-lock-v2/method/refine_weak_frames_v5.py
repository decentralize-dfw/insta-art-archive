from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from analyze_cv import CONFIG
OUT=Path(__file__).resolve().parent/'v2';cv2.setNumThreads(4);slug='board-game';M=np.load(OUT/f'{slug}-content-matrices-v3.npy');original=M.copy();c=CONFIG[slug];D=np.diag([1299.,1039.,1.]);Di=np.linalg.inv(D);sift=cv2.SIFT_create(4000,contrastThreshold=.012,edgeThreshold=18);clahe=cv2.createCLAHE(2,(8,8));mask=np.zeros((1040,1300),np.uint8);mask[25:-25,25:-25]=255
records=json.loads((OUT/f'{slug}-direct-content-v3.json').read_text())['per_frame'];selected=list(range(80,181))+list(range(365,401));anchors=[k for k in range(75,420,15) if records[k]['accepted']];need=sorted(set(selected)|set(anchors));cap=cv2.VideoCapture(str(c['src']));features={};tic=time.time()
for k in need:
 cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,im=cap.read();assert ok;im=cv2.warpPerspective(im,D@M[k],(1300,1040));g=clahe.apply(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY));kp,d=sift.detectAndCompute(g,mask);features[k]=(np.float32([p.pt for p in kp]),d)
cap.release();print('NATIVE FEATURES',len(features),round(time.time()-tic,1),flush=True);matcher=cv2.BFMatcher(cv2.NORM_L2);updated=[]
for k in selected:
 p,desc=features[k];targets={}
 if desc is None:continue
 for j in sorted([a for a in anchors if a!=k],key=lambda a:abs(a-k))[:4]:
  q,ref=features[j]
  if ref is None:continue
  for a,b in matcher.knnMatch(desc,ref,k=2):
   if a.distance<.7*b.distance:targets.setdefault(a.queryIdx,[]).append(q[a.trainIdx])
 aa=[];bb=[]
 for i,values in targets.items():
  values=np.array(values);target=np.median(values,axis=0)
  if len(values)>1 and np.median(np.linalg.norm(values-target,axis=1))>3:continue
  aa.append(p[i]);bb.append(target)
 if len(aa)<12:continue
 a=np.float32(aa);b=np.float32(bb);h,inl=cv2.findHomography(a,b,cv2.USAC_MAGSAC,1.5,maxIters=3000,confidence=.999)
 if h is None:continue
 use=inl.ravel().astype(bool)
 if use.sum()<16 or cv2.contourArea(cv2.convexHull(a[use]))/(1300*1040)<.2:continue
 selected_indices=[];cells={}
 for idx in np.where(use)[0]:
  cell=tuple((a[idx]/[1300,1040]*5).astype(int))
  if cells.get(cell,0)<8:selected_indices.append(idx);cells[cell]=cells.get(cell,0)+1
 if len(selected_indices)<8:continue
 h,_=cv2.findHomography(a[selected_indices],b[selected_indices],0)
 if h is None:continue
 h/=h[2,2];q=cv2.perspectiveTransform(np.float32([[[0,0],[1299,0],[1299,1039],[0,1039]]]),h)[0];shift=np.max(np.linalg.norm(q-[[0,0],[1299,0],[1299,1039],[0,1039]],axis=1))
 if shift>15:continue
 M[k]=Di@h@D@M[k];updated.append({'frame':k,'inliers':int(use.sum()),'max_corner_correction_px':float(shift)})
np.save(OUT/f'{slug}-content-matrices-v5.npy',M);np.save(OUT/f'{slug}-dense-matrices.npy',M);(OUT/f'{slug}-native-refinement-v5.json').write_text(json.dumps({'updated_frames':updated,'seconds':time.time()-tic},indent=2));print('NATIVE UPDATED',len(updated),round(time.time()-tic,1),flush=True)
