from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from analyze_cv import CONFIG
from register_content_v2 import SIFT,gray
from pose_graph_v2 import BASE,match
cv2.setNumThreads(4);OUT=Path(__file__).resolve().parent/'v2'
def main(slug):
 c=CONFIG[slug];cache=OUT/f'{slug}-independent-pairs.npz';base=BASE[slug];lo=base.min(axis=0);span=base.max(axis=0)-lo
 if not cache.exists():
  cap=cv2.VideoCapture(str(c['src']));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));sf=min(1.,1000/width);pairs=[(k,k+d) for k in range(7,n,15) for d in [3,33] if k+d<n];need=set(i for p in pairs for i in p);feats={};tic=time.time()
  for k in range(n):
   ok,im=cap.read();assert ok
   if k not in need:continue
   im=cv2.resize(im,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA);g=gray(im);mask=np.zeros_like(g);cv2.fillConvexPoly(mask,(base*sf).astype('int32'),255);mask=cv2.erode(mask,np.ones((15,15),np.uint8));kp,d=SIFT.detectAndCompute(g,mask)
   feats[k]={'xy':np.float32([p.pt for p in kp]).reshape(-1,2),'d':d if d is not None else np.empty((0,128),np.float32)}
  cap.release();I=[];J=[];A=[];B=[];counts=[]
  for i,j in pairs:
   r=match(feats[i],feats[j],sf,base*sf)
   if r is None:continue
   a,b,spread=r
   if len(a)<15 or spread<.25:continue
   I.extend([i]*len(a));J.extend([j]*len(a));A.extend(a/sf);B.extend(b/sf);counts.append([i,j,len(a),spread])
  np.savez(cache,I=I,J=J,A=A,B=B,counts=counts);print('INDEPENDENT MATCHES READY',slug,len(counts),'pairs',len(I),'points',round(time.time()-tic,1),'seconds',flush=True)
 matrices=OUT/f'{slug}-dense-matrices.npy'
 if not matrices.exists():print('Waiting for dense matrices',flush=True);return
 z=np.load(cache);I=z['I'].astype(int);J=z['J'].astype(int);A=np.c_[z['A'],np.ones(len(I))];B=np.c_[z['B'],np.ones(len(J))];M=np.load(matrices);size=np.array([1300.,1040.]) if slug=='board-game' else np.array([616.,840.])
 def transform(P,H):
  t=np.einsum('nij,nj->ni',H,P);return t[:,:2]/t[:,2:3]
 after=np.linalg.norm((transform(A,M[I])-transform(B,M[J]))*size,axis=1);before=np.linalg.norm((z['A']-z['B'])/span*size,axis=1);rows=[]
 for i,j,count,spread in z['counts']:
  use=(I==i)&(J==j);rows.append({'from_frame':int(i),'to_frame':int(j),'matches':int(count),'spread':float(spread),'before_median_px':float(np.median(before[use])),'after_median_px':float(np.median(after[use])),'after_p90_px':float(np.percentile(after[use],90))})
 summary={'independent_frame_pairs':len(rows),'independent_matched_points':len(I),'before_median_point_error_output_px':float(np.median(before)),'after_median_point_error_output_px':float(np.median(after)),'before_p95_pair_median_output_px':float(np.percentile([r['before_median_px'] for r in rows],95)),'after_p95_pair_median_output_px':float(np.percentile([r['after_median_px'] for r in rows],95)),'worst_pairs':sorted(rows,key=lambda r:r['after_median_px'],reverse=True)[:15]}
 (OUT/f'{slug}-independent-motion.json').write_text(json.dumps({'summary':summary,'pairs':rows},indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('slug',choices=list(CONFIG));main(p.parse_args().slug)
