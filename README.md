
# 📃 프로젝트 개요

<img width="809" alt="Untitled" src="https://user-images.githubusercontent.com/54202082/201047270-6071f002-3fbc-4eb9-b751-8bed3ffcfbe6.png">


COVID-19의 확산으로 우리나라를 비롯한 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었다. COVID-19의 감염 확산을 방지하기 위해서 마스크로 코와 입을 가리고 혹시 모를 감염자로부터의 전파 경로를 차단하는 것이 중요하다. 마스크는 코와 입을 완전히 가리도록 올바르게 착용했을 때 효과가 있다. 하지만 넓은 공공장소에서 모든 사람들의 마스크를 올바르게 착용했는지 검사하는 것은 추가적인 인적자원이 필요하다.

따라서 **카메라로 비춰진 사람 얼굴 이미지**만으로 이 사람이 **마스크를 쓰고 있는지, 쓰지 않았는지, 정확하게 쓴 것이 맞는지** 자동으로 가려낼 수 있는 시스템이 필요하다.

</br>

# 🗂️ 데이터셋

- **데이터의 개수** : Train 18,900장 / Test 12,600장

- **이미지 크기** : (384, 512)

- **클래스의 분류**

  - Mask : Wear, Incorrect, Not Wear
  - Gender : Male, Female
  - Age : <30 (Young), ≥30 and <60 (Middle), ≥60 (Old)
  
  </br>

  | Class |   Mask    | Gender |     Age      |
  | :---: | :-------: | :----: | :----------: |
  |   0   |   Wear    |  Male  |     <30      |
  |   1   |   Wear    |  Male  | >=30 and <60 |
  |   2   |   Wear    |  Male  |     >=60     |
  |   3   |   Wear    | Female |     <30      |
  |   4   |   Wear    | Female | >=30 and <60 |
  |   5   |   Wear    | Female |     >=60     |
  |   6   | Incorrect |  Male  |     <30      |
  |   7   | Incorrect |  Male  | >=30 and <60 |
  |   8   | Incorrect |  Male  |     >=60     |
  |   9   | Incorrect | Female |     <30      |
  |  10   | Incorrect | Female | >=30 and <60 |
  |  11   | Incorrect | Female |     >=60     |
  |  12   | Not Wear  |  Male  |     <30      |
  |  13   | Not Wear  |  Male  | >=30 and <60 |
  |  14   | Not Wear  |  Male  |     >=60     |
  |  15   | Not Wear  | Female |     <30      |
  |  16   | Not Wear  | Female | >=30 and <60 |
  |  17   | Not Wear  | Female |     >=60     |

</br>

# 🏆 프로젝트 결과

backbone을 서로 다른 조건으로 학습한 상위 5개 모델의 inference결과를 hard voting하여 제출한 inference 결과가 가장 높은 score를 달성.

- **앙상블에 사용한 모델**


  | idx  |                 model                 | f1-score | accuracy |
  | :--: | :-----------------------------------: | :------: | :------: |
  |  1   |         vit_base_patch16_224          |   0.74   |  78.38   |
  |  2   |     swin_base_patch4_window7_224      |   0.71   |  76.47   |
  |  3   |            efficientnet_b4            |   0.71   |  77.88   |
  |  4   |             custom model              |   0.70   |  77.71   |
  |  5   | efficientnet_b4 + RetinaFace + CutMix |   0.70   |  75.92   |

- **최종 결과**

  f1-score : 0.7436 | accuracy : 79.87
  
  ![Untitled 1](https://user-images.githubusercontent.com/54202082/201047328-3608cbf7-3947-4c15-947d-406ce2c26f9e.png)

</br>

# 👨‍👨‍👧‍👧 팀 구성 및 역할

<table>
  <tr height="35px">
    <td align="center" width="180px">
      <a href="https://github.com/iihye">이혜진_T4177</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/SS-hj">이하정_T4173</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/brotherhoon-code">김형훈_T4064</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/iden2t">김동영_T4027</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/ydsong2">송영동_T4109</a>
    </td>
  </tr>
  <tr height="35px">
    <td align="center" width="180px">
      <a> Data Analysis, Training </a>
    </td>
    <td align="center" width="180px">
      <a> Data Processing </a>
    </td>
    <td align="center" width="180px">
      <a> Modeling </a>
    </td>
    <td align="center" width="180px">
      <a> Training </a>
    </td>
    <td align="center" width="180px">
      <a> Training </a>
    </td>
  </tr>
</table>
