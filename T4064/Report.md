
# 2022-10-30  
## 베이스라인 모델(efficient) 성능과 에러 비율  

```
Epoch[30/30](414/414) || training loss 0.0039 || training acc 99.89% || train f1 0.9970 || lr 0.00012
Calculating validation results...
[30epoch Val] f1 : 0.9777, loss: 0.0304, acc: 96.40% || best f1 : 0.9846, best loss: 0.0298, best acc: 96.47%
generalization gap => f1 : 0.0124, current loss : -0.0265

2022-10-30 10:45
0.6332
```  

```
correct       5470
age             13
mask_state      11
gender           6
dtype: int64
```

왜 갑자기 퍼포먼스가 저하되었는지는 모르겠음.

## coatnet 성능과 에러비율
```
Epoch[30/30](414/414) || training loss 0.0310 || training acc 99.04% || train f1 0.9797 || lr 0.00014
Calculating validation results...
New best model for val f1 : 0.9769! saving the best model..
[30epoch Val] f1 : 0.9769, loss: 0.0467, acc: 95.77% || best f1 : 0.9769, best loss: 0.0467, best acc: 95.77%
generalization gap => f1 : 0.0027, current loss : -0.0157
```

```
correct       5430
age             47
gender          13
mask_state       9
gender,age       1
dtype: int64
```
별로 좋지가 않음  

## swin tf 성능과 에러 비율
```
Epoch[30/30](414/414) || training loss 0.0166 || training acc 99.54% || train f1 0.9934 || lr 0.00012
Calculating validation results...
New best model for val f1 : 0.9919! saving the best model..
[30epoch Val] f1 : 0.9919, loss: 0.0225, acc: 96.44% || best f1 : 0.9919, best loss: 0.0225, best acc: 96.51%
generalization gap => f1 : 0.0014, current loss : -0.0059
```

```
correct       5468
age             13
mask_state      13
gender           4
gender,age       2
dtype: int64
```

## comment  
* age는 다 별로 성능이 좋지 않고, 마스크 맞추는것은 eff, gender 맞추는것은 swin이 유리함.  
* 일단 coat은 갖다 버리자. convnext로 테스트해봄.  

## convnext 성능과 에러 비율  
```
Epoch[30/30](414/414) || training loss 0.0027 || training acc 99.92% || train f1 0.9986 || lr 0.00012
Calculating validation results...
New best model for val f1 : 0.9936! saving the best model..
[30epoch Val] f1 : 0.9936, loss: 0.0182, acc: 96.61% || best f1 : 0.9936, best loss: 0.0182, best acc: 96.61%
generalization gap => f1 : 0.0050, current loss : -0.0154

2022-10-30 20:13	
0.6921
```

```
correct       5478
mask_state      11
age              6
gender           4
gender,age       1
dtype: int64
```

## 3head, 2 classification + 1 regression으로 문제 변환  

성능이 불량함.
efficient net을 이용해서 문제를 푼 결과 아래와 같이 저조한 성적이 나옴.
```
Epoch[5/30](414/414) || train mask acc 71.44% || train gender acc 0.6141 || train age MAE 5.09
train acc 0.3749055177626606 || train f1 0.06398755903463277
Calculating validation results...
[5epoch Val] f1 : 0.0605, acc: 38.00% || best f1 : 0.0683, best acc: 38.00%
```

일단 3개 문제를 독립적으로 가져가는 방법을 실험하려 함.  
1. convnext age regression model  
2. eff1 gender classification model  
3. convnext gender classification model  
근데 regression model 이 아니라 classification model을 사용하면 더 좋아질까?  
아니다, 오히려 threshold를 지정해줄 수 있도록 regression model로 가자.  

## ConvNext age regression performance  



