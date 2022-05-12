# ComputerVision_TermProject_2022

본 프로젝트는 [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/pdf/1708.02551)의 pytorch를 이용한 구현입니다.

<img src="https://user-images.githubusercontent.com/88313282/168066652-15097823-6b0e-40b2-b78f-c6d23661764f.png" width="700 " height="400">

Sementic Segmentation 및 Instance Segmentation을 하는 과정에서 논문에서 제시하는 Discriminative Loss Function을 이용하여 모델을 학습하고 추론했을 때의 성능을 측정하는 연구를 진행합니다. 따라서 모든 Segmentation에서 Discriminative Loss를 사용할 수 있게 됩니다. 이 프로젝트를 진행하기 위해서 사용한 Network는 LaneNet이며, backbone에 ResNet을 부착하였고 Instance Segmentation을 통해 Lane Detection을 진행하도록 설계하였습니다. 
* Segmentation을 위한 베이스라인 모델 -> https://github.com/harryhan618/LaneNet

## Data Preperation
데이터셋은 차선을 인식하기 위한 Tusimple 데이터 셋을 사용하였으며, 데이터 트리의 구조는 다음과 같습니다. 또한 데이터는 [여기](https://github.com/TuSimple/tusimple-benchmark/issues/3)서 받을 수 있습니다.
```
Train_set
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
└── test_label.json

Test_set
├── clips

```

## Usage
1. 가상환경 설정
```
conda create -n seg-dis-loss python=3.8
conda activate seg-dis-loss
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requiremnets.txt
```
2. Training code
```
python train.py --data_dir /{path for train_set}
```
* 인자에 맞게 데이터 경로 및 모델, 체크포인트 등을 설정해줄 수 있습니다.
3. Test code
```
python test.py --data_dir /{path for test_set}
```
* 인자에 맞게 데이터 경로 및 모델, 체크포인트 등을 설정해줄 수 있습니다.

## Metric
- ACC (정확도)
- FP (False Positive)
- FN (False Negative)
- FPS (speed of forward pass)

## Result

|Architecture|Accuracy|FP|FN|FPS|
|-|-|-|-|-|
|Lane + Res18|0.940|0.142|0.085|15.6|
