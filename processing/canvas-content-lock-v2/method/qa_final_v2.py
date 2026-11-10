from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from PIL import Image,ImageDraw
from render_content_v2 import NAMES,RECTS
OUT=Path(__file__).resolve().parent/'v2'
for slug in NAMES:
 z=np.load(OUT/f'{slug}-independent-pairs.npz');M=np.load(OUT/f'{slug}-content-matrices-v3.npy');size=np.array(RECTS[slug][2:]);rows=[]
 def project(p,h):
  a=np.c_[p,np.ones(len(p))]@h.T;return a[:,:2]/a[:,2:3]*size
 for i,j,count,spread in z['counts']:
  i,j=int(i),int(j);use=(z['I']==i)&(z['J']==j);a=project(z['A'][use],M[i]);b=project(z['B'][use],M[j]);h,inl=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000,refineIters=15)
  if h is None:continue
  shift=np.c_[a,np.ones(len(a))]@h.T-a;rows.append({'from':i,'to':j,'coherent_median_px':float(np.median(np.linalg.norm(shift,axis=1))),'coherent_p90_px':float(np.percentile(np.linalg.norm(shift,axis=1),90))})
 report={'median_coherent_motion_px':float(np.median([r['coherent_median_px'] for r in rows])),'p95_coherent_motion_px':float(np.percentile([r['coherent_median_px'] for r in rows],95)),'worst':sorted(rows,key=lambda r:r['coherent_median_px'],reverse=True)[:12]}
 (OUT/f'{slug}-coherent-motion.json').write_text(json.dumps(report,indent=2));print(slug,json.dumps(report),flush=True)
 path=OUT/NAMES[slug]
 if not path.exists():continue
 cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));x,y,w,h=RECTS[slug];tw=360;th=round(h/w*tw);sheet=Image.new('RGB',(4*tw,3*(th+26)),'white');draw=ImageDraw.Draw(sheet)
 for k,idx in enumerate(np.linspace(2,n-1,12).astype(int)):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read()
  if not ok:continue
  im=im[y:y+h,x:x+w];im=cv2.resize(im,(tw,th),interpolation=cv2.INTER_AREA)
  for xx in [int(tw*.25),int(tw*.5),int(tw*.75)]:cv2.line(im,(xx,0),(xx,th-1),(0,255,0),1)
  for yy in [int(th*.25),int(th*.5),int(th*.75)]:cv2.line(im,(0,yy),(tw-1,yy),(0,255,0),1)
  px=(k%4)*tw;py=(k//4)*(th+26);sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(px,py));draw.text((px+4,py+th+5),f'{idx/30:.2f}s / frame {idx}',fill='black')
 cap.release();sheet.save(OUT/f'{slug}-FINAL-contact.jpg')
