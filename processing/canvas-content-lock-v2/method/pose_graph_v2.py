from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'));sys.path.insert(0,str(ROOT/'tmp/stabilization-v2-deps'))
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from analyze_cv import CONFIG
from register_content_v2 import SIFT,gray
cv2.setNumThreads(4);OUT=Path(__file__).resolve().parent/'v2'
BASE={'board-game':np.float32([[502,132],[1796,132],[1796,1166],[502,1166]]),'sept-neuf':np.float32([[142,29],[759,30],[761,884],[140,884]])}
def build(slug):
 c=CONFIG[slug];p=OUT/f'{slug}-graph-features-v3.npz'
 if p.exists():
  z=np.load(p);return z['frames'],z['scale'].item(),[{'xy':z[f'p{i}'],'d':z[f'd{i}']} for i in range(len(z['frames']))]
 cap=cv2.VideoCapture(str(c['src']));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));sf=min(1.,1000/w)
 samples=json.loads((OUT.parent/f'{slug}-registration-samples.json').read_text());extra=[s['frame'] for s in samples if s['result'] and s['result']['inliers']>=30 and s['result']['spread']>=.25 and s['result']['median_error']<3]
 frames=np.unique(np.r_[np.arange(0,n,15),n-1,extra]);data=[];cache={'frames':frames,'scale':sf};tic=time.time()
 for i,k in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(k));ok,im=cap.read();assert ok;im=cv2.resize(im,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA);g=gray(im)
  mask=np.zeros_like(g);q=BASE[slug]*sf;cv2.fillConvexPoly(mask,q.astype('int32'),255);mask=cv2.erode(mask,np.ones((15,15),np.uint8))
  kp,d=SIFT.detectAndCompute(g,mask);xy=np.float32([k.pt for k in kp]).reshape(-1,2)
  if d is None:d=np.empty((0,128),np.float32)
  data.append({'xy':xy,'d':d});cache[f'p{i}']=xy;cache[f'd{i}']=d
  if i%20==0:print('FEATURES',slug,i,'/',len(frames),round(time.time()-tic,1),flush=True)
 cap.release();np.savez(p,**cache);return frames,sf,data
def match(a,b,sf,base):
 if min(len(a['d']),len(b['d']))<12:return None
 pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(a['d'],b['d'],k=2);good=[x for x,y in pairs if x.distance<.73*y.distance]
 if len(good)<12:return None
 aa=a['xy'][[m.queryIdx for m in good]];bb=b['xy'][[m.trainIdx for m in good]]
 h,inl=cv2.findHomography(aa,bb,cv2.USAC_MAGSAC,1.5,maxIters=2500,confidence=.999)
 if h is None:return None
 use=inl.ravel().astype(bool);aa=aa[use];bb=bb[use]
 if len(aa)<12:return None
 spread=cv2.contourArea(cv2.convexHull(aa))/cv2.contourArea(base)
 if spread<.18:return None
 # Spatially balanced cap: a single detailed hand/face cannot supply all votes.
 lo=base.min(axis=0);span=base.max(axis=0)-lo;bins={};selected=[]
 for i,p in enumerate(aa):
  cell=tuple(np.clip(((p-lo)/span*5).astype(int),0,4))
  if bins.get(cell,0)<4:selected.append(i);bins[cell]=bins.get(cell,0)+1
 aa=aa[selected];bb=bb[selected]
 if len(aa)<10:return None
 return aa,bb,spread
def optimize(slug):
 c=CONFIG[slug];frames,sf,features=build(slug);n=len(frames);base=BASE[slug]*sf;lo=base.min(axis=0);span=base.max(axis=0)-lo;edges=[];tic=time.time()
 cache=OUT/f'{slug}-graph-edges-v3.npz'
 if cache.exists():
  z=np.load(cache);I,J,A,B=z['I'],z['J'],z['A'],z['B']
 else:
  ii=[];jj=[];aa=[];bb=[]
  for i in range(n):
   candidates=set(i+d for d in [1,2,3,5,8,12] if i+d<n);candidates.add(n-1)
   if slug=='board-game' and frames[i]<90:candidates.update(range(max(0,n-7),n))
   for j in sorted(candidates):
    result=match(features[i],features[j],sf,base)
    if result is None:continue
    a,b,spread=result;a=(a-lo)/span;b=(b-lo)/span
    ii.extend([i]*len(a));jj.extend([j]*len(a));aa.extend(a);bb.extend(b);edges.append((i,j,len(a),spread))
   if i%15==0:print('EDGES',slug,i,'/',n,'pairs',len(edges),flush=True)
  I=np.array(ii,int);J=np.array(jj,int);A=np.c_[np.array(aa),np.ones(len(aa))];B=np.c_[np.array(bb),np.ones(len(bb))];np.savez(cache,I=I,J=J,A=A,B=B)
 # The final whole-painting view defines the coordinate system. Initial whole
 # painting stills in board-game are the same fixed coordinate system.
 fixed={n-1};fixed_H={n-1:np.eye(3)}
 if slug=='board-game':fixed.update(i for i,k in enumerate(frames) if k<=75 or k>=750)
 else:fixed.add(0)
 fixed_H.update({i:np.eye(3) for i in fixed})
 # Prepared whole-canvas stage images share the completed painting's physical
 # four-corner coordinates. These observations anchor absolute scale/keystone
 # across progress stages instead of letting an image graph drift globally.
 for s in json.loads((OUT.parent/f'{slug}-registration-samples.json').read_text()):
  r=s['result']
  if r is None or r['inliers']<30 or r['spread']<.25 or r['median_error']>=3:continue
  node=int(np.searchsorted(frames,s['frame']));q=(np.float32(r['quad'])-BASE[slug].min(axis=0))/(BASE[slug].max(axis=0)-BASE[slug].min(axis=0))
  fixed.add(node);fixed_H[node]=cv2.getPerspectiveTransform(q.astype('float32'),np.float32([[0,0],[1,0],[1,1],[0,1]]))
 free=[i for i in range(n) if i not in fixed];index=np.full(n,-1,int);index[free]=np.arange(len(free));ident=np.array([1,0,0,0,1,0,0,0.],float);x0=np.tile(ident,(len(free),1));pixels=np.array([c['rw'],c['rh']],float)
 def matrices(x):
  params=np.tile(ident,(n,1));params[free]=x.reshape(-1,8);H=np.c_[params,np.ones(n)].reshape(n,3,3)
  for i,h in fixed_H.items():H[i]=h
  return H
 def projected(H,idx,P):
  z=np.einsum('nij,nj->ni',H[idx],P);return z[:,:2]/z[:,2:3],z[:,2]
 def fun(x):
  H=matrices(x);u,_=projected(H,I,A);v,_=projected(H,J,B);return np.r_[((u-v)*pixels).ravel(),(x-x0.ravel())*3.]
 m=len(I);pairrows=np.arange(m*2).reshape(m,2);prior_rows=2*m+np.arange(len(free)*8)
 def jac(x):
  H=matrices(x);row=[];col=[];vals=[]
  for ids,P,sgn in [(I,A,1.),(J,B,-1.)]:
   valid=index[ids]>=0;idx=np.nonzero(valid)[0];p=P[valid];uv,den=projected(H,ids[valid],p);xx,yy=p[:,0]/den,p[:,1]/den;one=1/den;u,v=uv[:,0],uv[:,1]
   du=np.c_[xx,yy,one,np.zeros((len(p),3)),-u*xx,-u*yy]*pixels[0]*sgn
   dv=np.c_[np.zeros((len(p),3)),xx,yy,one,-v*xx,-v*yy]*pixels[1]*sgn
   row.extend(np.repeat(pairrows[idx,0],8));row.extend(np.repeat(pairrows[idx,1],8));cc=index[ids[valid],None]*8+np.arange(8)[None];col.extend(cc.ravel());col.extend(cc.ravel());vals.extend(du.ravel());vals.extend(dv.ravel())
  row.extend(prior_rows);col.extend(np.arange(len(free)*8));vals.extend(np.full(len(free)*8,3.))
  return coo_matrix((vals,(row,col)),shape=(m*2+len(free)*8,len(free)*8)).tocsr()
 print('OPTIMIZE',slug,'nodes',n,'correspondences',m,'variables',len(free)*8,flush=True)
 result=least_squares(fun,x0.ravel(),jac=jac,loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=100,ftol=1e-6,xtol=1e-7,gtol=1e-6,verbose=1)
 H=matrices(result.x);np.savez(OUT/f'{slug}-optimized-anchors-v3.npz',frames=frames,H=H,base=BASE[slug],scale=sf)
 r=fun(result.x)[:m*2].reshape(-1,2);err=np.linalg.norm(r,axis=1);info={'frames':frames.tolist(),'fixed_nodes':sorted(fixed),'matches':m,'pairwise_world_error_median_px':float(np.median(err)),'pairwise_world_error_p90_px':float(np.percentile(err,90)),'cost':result.cost,'seconds':time.time()-tic,'success':bool(result.success),'message':result.message}
 (OUT/f'{slug}-pose-graph-report.json').write_text(json.dumps(info,indent=2),encoding='utf8');print('GRAPH DONE',slug,info,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('slug',choices=list(CONFIG));a=parser.parse_args();optimize(a.slug)
