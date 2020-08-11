#保证movirgraph_datast.py里
#assert len(boxes_and_labels) == len(self._image_paths)
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.datasets.moviegraph_dataset import *
from slowfast.datasets.moviegraph_helper import *
import os

args = parse_args()
cfg = load_config(args)

(
    image_paths,
    video_idx_to_name,
) = load_image_lists(cfg, is_train=False)

# Loading annotations for boxes and labels.
boxes_and_labels = load_boxes_and_labels(
    cfg, mode='val'
)

print('len(boxes_and_labels)',len(boxes_and_labels)) #9070
print('len(image_paths)',len(image_paths)) #9311
# assert len(boxes_and_labels) == len(image_paths)


# boxes_and_labels = [
#     boxes_and_labels[video_idx_to_name[i]]
#     for i in range(len(image_paths))
# ]

# # Get indices of keyframes and corresponding boxes and labels.
# (
#     keyframe_indices,
#     keyframe_boxes_and_labels,
# ) = moviegraph_helper.get_keyframe_data(boxes_and_labels)


g = open('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val_update.csv','w',encoding='utf-8')
with open('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val.csv','r') as f:
    data = f.readlines()
    for line in data:
        if line.split()[0] in boxes_and_labels.keys():
            g.write(line)
g.close()

os.rename('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val.csv','/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val.csv_9311')
os.rename('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val_update.csv','/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val.csv')