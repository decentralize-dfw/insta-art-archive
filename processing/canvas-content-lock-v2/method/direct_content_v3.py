from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from analyze_cv import CONFIG
from register_content_v2 import SIFT,gray
from pose_graph_v2 import BASE
cv2.setNumThreads(4);OUT=Path(__file__).resolve().parent/'v2'
def project(P,H):
 p=np.c_[P,np.ones(len(P))]@H.T;return p[:,:2]/p[:,2:3]
def main(slug):
 c=CONFIG[slug];z=np.load(OUT/f'{slug}-optimized-anchors-v3.npz');frames=z['frames'];hs=z['H'];base=BASE[slug];lo=base.min(axis=0);span=base.max(axis=0)-lo;features=np.load(OUT/f'{slug}-graph-features-v3.npz');sf=float(features['scale']);world=np.array([c['rw'],c['rh']],float)
 refs=[]
 for i,H in enumerate(hs):
  pts=features[f'p{i}']/sf;desc=features[f'd{i}'];wp=project((pts-lo)/span,H);matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=40))
  if len(desc)>1:matcher.add([desc]);matcher.train()
  refs.append({'pts':pts,'desc':desc,'world':wp,'matcher':matcher})
 cap=cv2.VideoCapture(str(c['src']));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));Bnorm=np.array([[1/span[0],0,-lo[0]/span[0]],[0,1/span[1],-lo[1]/span[1]],[0,0,1.]])
 D=np.diag([world[0],world[1],1.]);Di=np.linalg.inv(D);records=[];matrices=[];tic=time.time();corners=np.float32([[[0,0],[1,0],[1,1],[0,1]]])
 for k in range(n):
  ok,frame=cap.read();assert ok
  i=max(0,min(len(frames)-2,int(np.searchsorted(frames,k,side='right')-1)));u=(k-frames[i])/(frames[i+1]-frames[i]);initial=hs[i]*(1-u)+hs[i+1]*u;initial/=initial[2,2];M=initial@Bnorm;accepted=False;count=0;spread=0.;median=None;mode='unobservable/reference pose'
  fixed=(slug=='board-game' and (k<=75 or k>=750)) or k in [0,n-1]
  if not fixed:
   small=cv2.resize(frame,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA);g=gray(small);mask=np.zeros_like(g);cv2.fillConvexPoly(mask,(base*sf).astype('int32'),255);mask=cv2.erode(mask,np.ones((11,11),np.uint8));kp,desc=SIFT.detectAndCompute(g,mask)
   if desc is not None and len(desc)>=8:
    pts=np.float32([p.pt for p in kp])/sf;targets={};candidates=np.argsort(np.abs(frames-k))[:5]
    for j in candidates:
     r=refs[j]
     if len(r['desc'])<2:continue
     pairs=r['matcher'].knnMatch(desc,k=2)
     for a,b in pairs:
      if a.distance<.72*b.distance:targets.setdefault(a.queryIdx,[]).append(r['world'][a.trainIdx])
    aa=[];bb=[]
    for idx,values in targets.items():
     values=np.array(values);target=np.median(values,axis=0)
     if len(values)>1 and np.median(np.linalg.norm((values-target)*world,axis=1))>4:continue
     aa.append((pts[idx]-lo)/span);bb.append(target)
    if len(aa)>=10:
     A=np.float32(aa);W=np.float32(bb);F,inliers=cv2.findHomography(A*world,W*world,cv2.USAC_MAGSAC,1.3,maxIters=3000,confidence=.999)
     if F is not None:
      use=inliers.ravel().astype(bool);count=int(use.sum());spread=cv2.contourArea(cv2.convexHull(A[use]));balanced=[];cells={}
      for t in np.where(use)[0]:
       cell=tuple(np.clip((A[t]*5).astype(int),0,4))
       if cells.get(cell,0)<6:balanced.append(t);cells[cell]=cells.get(cell,0)+1
      if len(balanced)>=12 and spread>.20:
       F,_=cv2.findHomography(A[balanced]*world,W[balanced]*world,0);H=Di@F@D
       if H is not None and np.isfinite(H).all():
        q=project(corners[0],np.linalg.inv(H));area=cv2.contourArea(q.astype('float32'));err=np.linalg.norm((project(A[use],H)-W[use])*world,axis=1);median=float(np.median(err))
        if cv2.isContourConvex(q.astype('float32')) and .55<area<1.65 and np.max(np.abs(q-corners[0]))<.22 and median<1.5:
         M=H@Bnorm;accepted=True;mode='direct stable-paint registration'
  else:
   if k in frames:M=hs[int(np.where(frames==k)[0][0])]@Bnorm
   mode='completed-painting fixed reference' if slug=='board-game' else 'physical four-corner endpoint reference'
  matrices.append(M);records.append({'frame':k,'accepted':accepted or fixed,'mode':mode,'inliers':count,'spread':spread,'fit_median_world_px':median})
  if k%100==0:print('DIRECT',slug,k,'/',n,'accepted',sum(r['accepted'] for r in records),'seconds',round(time.time()-tic,1),flush=True)
 cap.release();np.save(OUT/f'{slug}-content-matrices-v3.npy',np.array(matrices));np.save(OUT/f'{slug}-dense-matrices.npy',np.array(matrices));(OUT/f'{slug}-direct-content-v3.json').write_text(json.dumps({'frames':n,'accepted_frames':sum(r['accepted'] for r in records),'seconds':time.time()-tic,'per_frame':records}),encoding='utf8');print('DIRECT DONE',slug,round(time.time()-tic,1),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('slug',choices=list(CONFIG));main(p.parse_args().slug)
