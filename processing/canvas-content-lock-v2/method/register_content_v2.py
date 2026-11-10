from pathlib import Path
import sys,json,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from analyze_cv import CONFIG,read_image
cv2.setNumThreads(4)
OUT=Path(__file__).resolve().parent/'v2';OUT.mkdir(exist_ok=True)
SIFT=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.012,edgeThreshold=12)
CLAHE=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
def gray(im):return CLAHE.apply(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
def refs(c,slug):
 dest=OUT/f'{slug}-reference-features.npz'
 if dest.exists():
  z=np.load(dest);return [{'n':n,'points':z[f'p{n}'],'desc':z[f'd{n}']} for n in c['nums']]
 result=[];cache={}
 for n in c['nums']:
  im=read_image(c['refs']/f'{n}.png');im=cv2.resize(im,(c['rw'],c['rh']),interpolation=cv2.INTER_AREA)
  kp,d=SIFT.detectAndCompute(gray(im),None);p=np.float32([k.pt for k in kp]);result.append({'n':n,'points':p,'desc':d});cache[f'p{n}']=p;cache[f'd{n}']=d
 np.savez(dest,**cache);return result
def fit_regularized(a,b,base,c,scale):
 # Pixel matches determine the pose. Very weak corner observations prevent
 # unobserved corners exploding when an arm covers most of the surface.
 rw,rh=c['rw'],c['rh'];centre=base.mean(axis=0);span=np.array([np.ptp(base[:,0]),np.ptp(base[:,1])])
 aa=(a/np.array([rw,rh])-.5)*2;bb=(b-centre)/span*2
 ca=np.float64([[-1,-1],[1,-1],[1,1],[-1,1]]);cb=(base-centre)/span*2
 aa=np.vstack([aa,ca]);bb=np.vstack([bb,cb]);weights=np.r_[np.ones(len(a)),np.full(4,.10)]
 x,y=aa.T;u,v=bb.T;one=np.ones(len(aa));zero=np.zeros(len(aa))
 A=np.empty((len(aa)*2,8));A[0::2]=np.array([x,y,one,zero,zero,zero,-u*x,-u*y]).T;A[1::2]=np.array([zero,zero,zero,x,y,one,-v*x,-v*y]).T
 target=bb.ravel();ww=np.repeat(np.sqrt(weights),2);h=np.r_[np.linalg.lstsq(A*ww[:,None],target*ww,rcond=None)[0],1].reshape(3,3)
 srcnorm=np.array([[2/rw,0,-1],[0,2/rh,-1],[0,0,1.]])
 dstnorm=np.array([[2/span[0],0,-centre[0]*2/span[0]],[0,2/span[1],-centre[1]*2/span[1]],[0,0,1.]])
 return np.linalg.inv(dstnorm)@h@srcnorm
def register(frame,c,rr,candidates):
 sf=min(1.,1100/frame.shape[1]);small=cv2.resize(frame,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA)
 kp,desc=SIFT.detectAndCompute(gray(small),None)
 if desc is None:return None
 pts=np.float32([p.pt for p in kp]);matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=40));matcher.add([desc]);matcher.train();results=[]
 x0,y0,x1,y1=c['rough'];base=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])*sf
 corners=np.float32([[[0,0],[c['rw']-1,0],[c['rw']-1,c['rh']-1],[0,c['rh']-1]]])
 for n in candidates:
  r=rr[n-1];pairs=matcher.knnMatch(r['desc'],k=2);good=[m for m,l in pairs if m.distance<.76*l.distance]
  if len(good)<16:continue
  a=r['points'][[m.queryIdx for m in good]];b=pts[[m.trainIdx for m in good]]
  h,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,1.5,maxIters=3000,confidence=.999)
  if h is None:continue
  mask=mask.ravel().astype(bool);a=a[mask];b=b[mask]
  if len(a)<16:continue
  coverage=cv2.contourArea(cv2.convexHull(a))/(c['rw']*c['rh'])
  if coverage<.14:continue
  # Held-out correspondences are never used in the fitted transform.
  order=np.lexsort((a[:,1],a[:,0]));hold=order[::5];train=np.setdiff1d(np.arange(len(a)),hold)
  h=fit_regularized(a[train],b[train],base,c,sf)
  q=cv2.perspectiveTransform(corners,h)[0];proj=cv2.perspectiveTransform(a[None],h)[0];errors=np.linalg.norm(proj-b,axis=1)/sf
  area=cv2.contourArea(q);expect=cv2.contourArea(base)
  if not cv2.isContourConvex(q) or not .70*expect<area<1.35*expect:continue
  if np.max(np.linalg.norm(q-base,axis=1))/sf>max(frame.shape[:2])*.09:continue
  if np.median(errors[hold])>2.5 or np.percentile(errors[hold],90)>5:continue
  result={'ref':n,'quad':(q/sf).tolist(),'matches':len(a),'spread':coverage,'holdout_median_px':float(np.median(errors[hold])),'holdout_p90_px':float(np.percentile(errors[hold],90)),'score':len(a)*min(1,coverage/.45)}
  results.append(result)
  if len(a)>100 and coverage>.4:break
 return max(results,key=lambda r:r['score']) if results else None
def main():
 parser=argparse.ArgumentParser();parser.add_argument('slug',choices=list(CONFIG));parser.add_argument('--start',type=int,default=0);parser.add_argument('--end',type=int);parser.add_argument('--step',type=int,default=1);args=parser.parse_args()
 slug=args.slug;c=CONFIG[slug];rr=refs(c,slug);samples=json.loads((OUT.parent/f'{slug}-registration-samples.json').read_text(encoding='utf8'));samples=[x for x in samples if x['result']]
 cap=cv2.VideoCapture(str(c['src']));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));end=args.end or total;start=time.time();records=[];last=None
 cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
 for k in range(args.start,end):
  ok,frame=cap.read()
  if not ok:break
  if (k-args.start)%args.step:continue
  closest=sorted(samples,key=lambda s:abs(s['frame']-k))[:2];ns=[s['result']['ref'] for s in closest]
  cand=[]
  for n in ([last] if last else [])+ns+[n+d for n in ns for d in [-1,1,-2,2]]:
   if n in c['nums'] and n not in cand:cand.append(n)
  r=register(frame,c,rr,cand)
  if r:last=r['ref']
  records.append({'frame':k,'result':r})
  if len(records)%30==0:
   (OUT/f'{slug}-{args.start}-{end}-registration.json').write_text(json.dumps(records),encoding='utf8')
   print(slug,k,'valid',sum(x['result'] is not None for x in records),'/',len(records),'seconds',round(time.time()-start,1),flush=True)
 cap.release();(OUT/f'{slug}-{args.start}-{end}-registration.json').write_text(json.dumps(records),encoding='utf8');print('DONE',slug,len(records),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
