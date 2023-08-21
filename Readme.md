paper for "RPU-PVB: Robust Object Detection Based on a Unified Metric Perspective with Bilinear Interpolation"



### Requirements
* Python 3.9
* PyTorch >= 1.8
* numpy
* cv2

```bash
pip install -r requirements.txt  # install requirements
```

### Data Preparation
Download the PASCAL VOC and MS-COCO dataset and unpack them. The data structure should look like this

**VOC**
```
VOCdevkit
|-- VOC2007
|      |-- Annotations
|      |-- ImageSets
|      |-- JPEGImages
|-- VOC2012
       |-- Annotations
       |-- ImageSets
       |-- JPEGImages
```

**COCO**
```
coco2017
|-- 2017_clean  # merge train2017 and val2017
|-- annotations
       |-- instances_train2017.json
       |-- instances_val2017.json
```

### RPU-PVB
We provide codes to reproduce the results in our paper.

#### Training
```bash
python train_robust.py --cfg cfgs/RobustDet_voc.yaml --adv_type mtd --data_use clean --multi_gpu False \
    --basenet weights/ssd300_mAP_77.43_v2.pth --dataset_root <path_to_your_VOC_root>
```

Training on COCO dataset:
```bash
python train_robust.py --cfg cfgs/RobustDet_coco.yaml --adv_type mtd --data_use clean --multi_gpu False \
    --basenet weights/ssd300_COCO_clean_final_300000.pth --dataset_root <path_to_your_COCO_root>
```

#### Evaluation
VOC
```bash
python eval_attack.py --cfg cfgs/RobustDet_voc.yaml --trained_model RPU-PVB_VOC.pth \
    --data_use clean --adv_type cls \ # attack type, choice in [clean, cls, loc, cwat, dag]
     --dataset_root <path_to_your_VOC_root>
```
```
You get the following results (with small deviations)
Clean	Acls	Aloc	CWA	DAG
73.0   60.2   57.5   60.9   63.2
```

COCO
```bash
python eval_attack.py --cfg cfgs/RobustDet_coco.yaml --trained_model RPU-PVB_COCO.pth \
    --data_use clean --adv_type cls \ # attack type, choice in [clean, cls, loc, cwat, dag]
     --dataset_root <path_to_your_COCO_root>
```
```
You get the following results (with small deviations)
Clean	Acls	Aloc	CWA	DAG
36.2 	24.5 	27.1 	25.3 	26.6 
```
## Pretrained Models
link：https://pan.baidu.com/s/1izE1r12NebhJKLP2GjDslw?pwd=obhh 
password：obhh
