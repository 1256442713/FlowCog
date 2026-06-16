# Scene Understanding and Counting

本项目实现未知监控场景下交通路网的提取和车道级车辆实时计数的功能。项目包含三个子模块

- 头端模块：多目标跟踪（MOT），生成车辆运动轨迹，根据车辆运动轨迹和交通路网进行车道级交通流量统计；将车辆的运动轨迹上传至云端模块
- 云端模块：当头端模块上传的车辆运动轨迹累积到一定数目的时候，进行轨迹聚类、道路建模、道路分类，最终更新交通路网发送至头端模块
- 通信模块：负责头端和云端的消息传输


## Getting Started

### Prerequisites

#### Clone this repository
```bash
git clone --recurse-submodules https://git.pcl.ac.cn/digital-retina-alg/traffic-UAC.git && cd traffic-UAC
``` 

#### Setup runtime environment
本项目运行在docker容器中，``docker``目录下提供了构建镜像的[Dockerfile](https://git.pcl.ac.cn/digital-retina-alg/traffic-UAC/src/branch/master/docker/Dockerfile)
```bash
# compile trajectory related modules
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# compile c modules and python extensions
make all
# test
make runtest
# install python dependencies
make init

# setup CenterTrack runtime
# if DCNv2 directory already exists, skip this git clone step
cd ../CenterTrack/src/lib/model/networks
git clone https://github.com/CharlesShang/DCNv2
# compile DCNv2 module
cd DCNv2
python setup.py build
python setup.py develop
```

## Usage

### Head module
```bash
cd $root
python head_app.py --data_path $video_path --tracker CenterTrack --track_thresh 0.4 \
    --load_model CenterTrack/exp/tracking/detrac_train/model_epoch_15.pth \
    --gpus 0
```
### Cloud module
#### Train

```bash
python train.py /data_dir --traj_sample_num 21
```

- `/data_dir` 指定存放训练文件的文件夹，其中训练文件为 .txt 文件
- `--traj_sample_num` 指定特征提取时对一条轨迹的采样数，默认设置21
- 训练结束后的模型文件保存在当前目录下的 `classification.pkl` 文件中


#### Validate

```
python validate.py /val_dir /model_path --sample_num 21
```

- `/val_dir` 指定验证集文件的文件夹，其中轨迹文件为 .txt 文件

#### Service

```bash
python server_app.py --log_dir /tmp/trafficUAC
```

- `--log_dir` 指定轨迹聚类模型的存放路径

#### Predict

```
python predict.py /test_dir /model_path --save out.bin --sample_num 21
```

- `/test_dir` 指定待预测的轨迹文件的文件夹，其中轨迹文件为 .bin 格式
- `/model_path` 指定模型文件
- `--save ` 指定存储预测及轨迹信息的二进制文件，默认为`out.bin` 文件
- `--traj_sample_num` 指定特征提取时对一条轨迹的采样数，默认设置21



