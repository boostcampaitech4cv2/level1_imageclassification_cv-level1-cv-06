# 김동영A T4027
## K-fold validation & ensemble 진행
5 fold validation
Soft voting

## Model
timm 의 'ViT_base_patch16_224' 모델 사용

## Data Preprocessing
Age band : young : < 30
         : middle : 30 < <57
         : old : 57 <

## Data Augmentation
k_fold_train.py 확인

## Fine-Tuning
Mask, Gender, Age 를 추정하는 Multi-label classification
### Loss
Mask, Gender : focal loss 사용
Age : label smoothing loss 사용
### Scheduler
ReduceLROnPlateau
