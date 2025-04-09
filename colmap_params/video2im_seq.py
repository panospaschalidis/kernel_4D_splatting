import cv2
import os
import argparse
import numpy as np 
import pdb

from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--sub_sample', action="store_true", default=False)
    
    args = parser.parse_args()
    if not args.video_path:
        print('Please provide a valid video path')
        raise NotImplementedError

    cam = cv2.VideoCapture(args.video_path)
    im_dir = os.path.splitext(os.path.basename(args.video_path))[0]
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path,'color_full'), exist_ok=True)
    frameno = 0
    while(True):
       ret,frame = cam.read()
      
       if ret:
          if frameno==0:
            height, width, _ = frame.shape
          # if video is still left continue creating images
          name = os.path.join(os.getcwd(),args.out_path,'color_full','frame_' +str(frameno).zfill(6) + '.png')
          print ('new frame captured...' + name)
          if args.sub_sample:
              frame = np.array(Image.fromarray(frame).resize((int(width/2), int(height/2))))
              cv2.imwrite(name, frame)
          else:
              cv2.imwrite(name, frame)
          frameno += 1
       else:
          break

    cam.release()
    cv2.destroyAllWindows()
    list_ = [str(frameno)+'\n', str(width)+'\n' , str(height)+'\n']
    frame_list = ['frame_'+str(i).zfill(6)+'\n' for i in range(frameno)]
    list_.extend(frame_list)
    with open(os.path.join(im_dir, 'frames.txt'), 'w') as f:
        f.writelines(list_)
    f.close()
        
