#using mmdet env

from mmdet.apis import init_detector, inference_detector
import mmcv
import os

THRESHOLD_SCORE = 0.8
config_file = '/data/wushiwei/projects/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco.py'
checkpoint_file = '/data/wushiwei/data/SRRMM_ann/checkpoints/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth'
frame_dir = '/data/wushiwei/data/moviegraph/1fps_version'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0,1,2,3')

with open('/data/wushiwei/data/SRRMM_ann/moviegraph/person_box_ann/val_predicted_box.csv','w') as f:
    # test a single image and show the results
    # img = '/data/wushiwei/data/moviegraph/1fps_version/tt0037884/scene-002.ss-0007.es-0009/000000.jpg'  # or img = mmcv.imread(img), which will only load it once
    for video in os.listdir(frame_dir):
        video_dir = os.path.join(frame_dir,video)
        for clip in os.listdir(video_dir):
            clip_dir = os.path.join(video_dir,clip)
            imgs = os.listdir(clip_dir)
            imgs.sort(key= lambda x:int(x[:-4]))
            if len(imgs) > 1:
                FPS = int(imgs[1][:-4])
                for img in imgs:
                    frame_time = int(int(img[:-4])/FPS)
                    img = os.path.join(clip_dir,img)
                    height,width = mmcv.imread(img).shape[:-1]
                    result = inference_detector(model, img)
                    if len(result[0]) > 0:
                        # visualize the results in a new window
                        # model.show_result(img, result)
                        # or save the visualization results to image files
                        # model.show_result(img, result, out_file='result.jpg')
                        person_box = result[0][0][:-1]
                        score = result[0][0][-1]
                        if score > THRESHOLD_SCORE:
                            # print(','.join((video + '_' + clip, str(frame_time), '%.3f' % (person_box[0]/width), '%.3f' % (person_box[1]/height), '%.3f' % (person_box[2]/width), '%.3f' % (person_box[3]/height), '%.6f' % (score))))
                            f.write(','.join((video + '_' + clip, str(frame_time), '%.3f' % (person_box[0]/width), '%.3f' % (person_box[1]/height), '%.3f' % (person_box[2]/width), '%.3f' % (person_box[3]/height), '%.6f' % (score))) + '\n')
            print(video + clip + 'finished')
        print(video + 'finished')
    # test a video and show the results
    # video = mmcv.VideoReader('video.mp4')
    # for frame in video:
    #     result = inference_detector(model, frame)
    #     model.show_result(frame, result, wait_time=1)
