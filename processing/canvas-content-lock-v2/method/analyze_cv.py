from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tmp/reference-analysis-tools'))
import cv2,numpy as np
cv2.setNumThreads(4)
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SIFT=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.018,edgeThreshold=15)
CONFIG={
 'board-game':{'src':ROOT/'RESİMLER DETAY/KOPSTAL/kopstal_6_7thsela/MERGED-20sec.mp4','refs':ROOT/'RESİMLER DETAY/KOPSTAL/kopstal_6_7thsela/İPHONE PROCESS/FRAMED','nums':list(range(1,15)),'rw':800,'rh':640,'rough':[500,130,1810,1168]},
 'sept-neuf':{'src':ROOT/'RESİMLER DETAY/KOPSTAL/kopstal_7_nine-eleven/KARŞIDAN-2_1.mp4','refs':ROOT/'RESİMLER DETAY/KOPSTAL/kopstal_7_nine-eleven/PROCESS ihpne','nums':list(range(1,19)),'rw':600,'rh':818,'rough':[130,20,775,885]}}
def read_image(path):
 im=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
 if im is None:raise RuntimeError(str(path))
 return im
def features(im):
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 return SIFT.detectAndCompute(gray,None)
def references(c):
 refs=[]
 for n in c['nums']:
  im=read_image(c['refs']/f'{n}.png');im=cv2.resize(im,(c['rw'],c['rh']),interpolation=cv2.INTER_AREA)
  kp,desc=features(im);refs.append({'n':n,'im':im,'kp':kp,'desc':desc})
 return refs
def register(frame,c,refs):
 factor=min(1,1100/frame.shape[1]);small=cv2.resize(frame,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
 kp,desc=features(small)
 if desc is None:return None
 matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=60));results=[]
 for r in refs:
  if r['desc'] is None:continue
  pairs=matcher.knnMatch(r['desc'],desc,k=2);good=[a for a,b in pairs if a.distance<.72*b.distance]
  if len(good)<9:continue
  a=np.float32([r['kp'][m.queryIdx].pt for m in good]);b=np.float32([kp[m.trainIdx].pt for m in good])
  h,inl=cv2.findHomography(a,b,cv2.USAC_MAGSAC,2.0,maxIters=4000,confidence=.999)
  if h is None:continue
  inl=inl.ravel().astype(bool);count=int(inl.sum())
  if count<9:continue
  corners=np.float32([[[0,0],[c['rw']-1,0],[c['rw']-1,c['rh']-1],[0,c['rh']-1]]])
  q=cv2.perspectiveTransform(corners,h)[0]/factor
  x0,y0,x1,y1=c['rough'];expected=(x1-x0)*(y1-y0);area=cv2.contourArea(q)
  if not cv2.isContourConvex(q) or not .55*expected<area<1.55*expected:continue
  if np.max(np.abs(q-np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])))>max(frame.shape)*.17:continue
  proj=cv2.perspectiveTransform(a[None],h)[0];err=np.linalg.norm(proj-b,axis=1)[inl]
  hull=cv2.contourArea(cv2.convexHull(a[inl]))/(c['rw']*c['rh'])
  if hull<.04:continue
  results.append({'ref':r['n'],'quad':q.tolist(),'inliers':count,'spread':hull,'median_error':float(np.median(err)/factor),'score':count*min(1,hull/.4)})
 return max(results,key=lambda r:r['score']) if results else None
def main():
 for slug,c in CONFIG.items():
  start=time.time();refs=references(c);cap=cv2.VideoCapture(str(c['src']));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);samples=[];tiles=[]
  for t in np.linspace(.6,(n-1)/fps,24):
   k=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,frame=cap.read()
   if not ok:continue
   found=register(frame,c,refs);samples.append({'frame':k,'t':k/fps,'result':found})
   if found:cv2.polylines(frame,[np.int32(found['quad'])],True,(0,0,255),3)
   cv2.putText(frame,f'{k/fps:.1f}s ref {found["ref"] if found else "NONE"} / {found["inliers"] if found else 0}',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
   tile=cv2.resize(frame,(460,260) if slug=='board-game' else (300,300));tiles.append(tile)
   print(slug,round(t,2),None if not found else (found['ref'],found['inliers'],round(found['spread'],2),round(found['median_error'],2)),flush=True)
  cap.release();height,width=tiles[0].shape[:2];sheet=np.full((((len(tiles)+3)//4)*height,width*4,3),255,np.uint8)
  for i,tile in enumerate(tiles):sheet[(i//4)*height:(i//4+1)*height,(i%4)*width:(i%4+1)*width]=tile
  cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,93])[1].tofile(str(OUT/f'{slug}-registered-samples.jpg'))
  (OUT/f'{slug}-registration-samples.json').write_text(json.dumps(samples,indent=2),encoding='utf8')
  print(slug,'analysis seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
