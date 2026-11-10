from pathlib import Path
import sys,json,time,subprocess,hashlib,csv,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tmp/reference-analysis-tools'))
import cv2,numpy as np
from analyze_cv import CONFIG
cv2.setNumThreads(4);OUT=Path(__file__).resolve().parent/'v2'
NAMES={'board-game':'BOARD-CONTENT-LOCK-V2.mp4','sept-neuf':'SEPT-CONTENT-LOCK-V2.mp4'}
RECTS={'board-game':(500,130,1300,1040),'sept-neuf':(142,30,616,840)}
def run(args):return subprocess.run(args,capture_output=True,check=True,encoding='utf8',errors='replace')
def main(slug):
 c=CONFIG[slug];matrices=np.load(OUT/f'{slug}-content-matrices-v3.npy');cap=cv2.VideoCapture(str(c['src']));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));x,y,w,h=RECTS[slug];D=np.diag([w-1.,h-1.,1.]);path=OUT/NAMES[slug]
 cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-i',str(c['src']),'-map','0:v:0','-map','1:a:0?','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709','-c:v','libx264','-preset','slow','-crf','17','-maxrate','25M','-bufsize','25M','-threads','4','-pix_fmt','yuv420p','-c:a','copy','-color_range','tv','-colorspace','bt709','-color_trc','bt709','-color_primaries','bt709','-movflags','+faststart',str(path)]
 log=open(OUT/f'{slug}-final-encode.log','w');encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log);start=time.time();rows=[];corners=np.float32([[[0,0],[1,0],[1,1],[0,1]]])
 for k,M in enumerate(matrices):
  ok,frame=cap.read();assert ok
  # This is exclusively a warp of the SAME source frame. Neither anchor
  # pictures nor temporal median images are ever composited into output.
  patch=cv2.warpPerspective(frame,D@M,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
  output=np.full_like(frame,255)
  if slug=='board-game':
   sample=np.concatenate([frame[10,::80],frame[-10,::80]],axis=0);output[:]=np.median(sample,axis=0).astype('uint8')
  output[y:y+h,x:x+w]=patch;encoder.stdin.write(output.tobytes());quad=cv2.perspectiveTransform(corners,np.linalg.inv(M))[0];rows.append([k,*quad.ravel()])
  if k%150==0:print('RENDER V2',slug,k,'/',len(matrices),round(time.time()-start,1),'seconds',flush=True)
 cap.release();encoder.stdin.close();assert encoder.wait()==0;log.close()
 np.savetxt(OUT/f'{slug}-four-corners-v2.csv',rows,delimiter=',',header='frame,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y',comments='',fmt=['%d']+['%.8f']*8)
 data=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]).stdout);v=next(s for s in data['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==len(matrices) and (v['width'],v['height'])==(width,height) and v['avg_frame_rate']=='30/1'
 run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-map','0:v:0','-map','0:a:0?','-f','null','-'])
 def audio(p):return run(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-f','hash','-hash','sha256','-']).stdout.strip()
 a,b=audio(c['src']),audio(path);assert a==b
 info={'file':path.name,'frames':len(matrices),'duration_seconds':len(matrices)/fps,'width':width,'height':height,'fps':fps,'audio_identical':True,'audio_hash':a,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'source_frame_replacement':False,'inpainting':False,'ai_fill':False,'missing_regions':'remain empty/black or retain the already-baked source matte','geometry':'one independent four-corner homography for every original frame, registered on painting content','decode_passed':True,'encode_seconds':time.time()-start}
 (OUT/f'{slug}-final-validation.json').write_text(json.dumps(info,indent=2),encoding='utf8');print('FINAL V2 READY',slug,info,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('slug',choices=list(CONFIG));main(p.parse_args().slug)
