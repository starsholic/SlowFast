import os

with open('val_raw.csv','w') as f:
    f.write('video_clip path' + '\n')
    data_path = '/data/wushiwei/data/moviegraph/1fps_version'
    for video in os.listdir(data_path):
        video_path = os.path.join(data_path,video)
        for clip in os.listdir(video_path):
            clip_path = os.path.join(video_path,clip)
            for img in os.listdir(clip_path):
                img_path = os.path.join(video,clip,img)
                f.write(video + '_' + clip + ' ' + img_path + '\n')
        print(video + 'finished')
print('all_finished')
