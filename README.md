# Multi-Task Learning project
Unofficial implimitation of Multi-task learning using uncertainty to weigh losses for scene geometry and semantics [[arXiv](https://arxiv.org/abs/1705.07115)].

## Results
### Examples
|        Input        | Label <br>segmentation  |Instance <br>segmentation|       Depth         |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|<img width="200px" src='inputs/Pedestrian_crossing_0.png'>|<img src='results/resNet_label_instance_disp/label_Pedestrian_crossing_0.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_Pedestrian_crossing_0.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_Pedestrian_crossing_0.png' width="200px">|
|<img width="200px" src='inputs/Pedestrian_crossing_1.png'>|<img src='results/resNet_label_instance_disp/label_Pedestrian_crossing_1.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_Pedestrian_crossing_1.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_Pedestrian_crossing_1.png' width="200px">|
|<img width="200px" src='inputs/bicycle_0.png'>|<img src='results/resNet_label_instance_disp/label_bicycle_0.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_bicycle_0.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_bicycle_0.png' width="200px">|
|<img width="200px" src='inputs/bicycle_1.png'>|<img src='results/resNet_label_instance_disp/label_bicycle_1.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_bicycle_1.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_bicycle_1.png' width="200px">|
|<img width="200px" src='inputs/bus_0.png'>|<img src='results/resNet_label_instance_disp/label_bus_0.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_bus_0.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_bus_0.png' width="200px">|
|<img width="200px" src='inputs/bus_1.png'>|<img src='results/resNet_label_instance_disp/label_bus_1.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_bus_1.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_bus_1.png' width="200px">|
|<img width="200px" src='inputs/parking_0.png'>|<img src='results/resNet_label_instance_disp/label_parking_0.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_parking_0.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_parking_0.png' width="200px">|
|<img width="200px" src='inputs/parking_1.png'>|<img src='results/resNet_label_instance_disp/label_parking_1.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_parking_1.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_parking_1.png' width="200px">|
|<img width="200px" src='inputs/truck_0.png'>|<img src='results/resNet_label_instance_disp/label_truck_0.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_truck_0.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_truck_0.png' width="200px">|
|<img width="200px" src='inputs/truck_1.png'>|<img src='results/resNet_label_instance_disp/label_truck_1.png' width="200px">|<img src='results/resNet_label_instance_disp/instance_truck_1.png' width="200px">|<img src='results/resNet_label_instance_disp/disp_truck_1.png' width="200px">|

### Single vs. duble vs. All: Task Result per Epoch
|Label segmentation   |Instance segmentation|       Depth         |
|:-------------------:|:-------------------:|:-------------------:|
|<img width="200px" src='images/graphs/label.png'>|<img width="200px" src='images/graphs/instance.png'>|<img width="200px" src='images/graphs/disp.png'>|



