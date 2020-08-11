#保证movirgraph_datast.py里
#assert len(boxes_and_labels) == len(self._image_paths)
import csv

g = open('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val_update.csv','w',encoding='utf-8')
with open('/data/wushiwei/data/SRRMM_ann/moviegraph/frame_list/val.csv','r') as f:
    data = f.readlines()
    for line in data:
        if line.split()[0] in all_boxes.keys():
            g.write(line)
g.close()