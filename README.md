
# π νλ‘μ νΈ κ°μ

<img width="809" alt="Untitled" src="https://user-images.githubusercontent.com/54202082/201047270-6071f002-3fbc-4eb9-b751-8bed3ffcfbe6.png">


COVID-19μ νμ°μΌλ‘ μ°λ¦¬λλΌλ₯Ό λΉλ‘―ν μ  μΈκ³ μ¬λλ€μ κ²½μ μ , μμ°μ μΈ νλμ λ§μ μ μ½μ κ°μ§κ² λμλ€. COVID-19μ κ°μΌ νμ°μ λ°©μ§νκΈ° μν΄μ λ§μ€ν¬λ‘ μ½μ μμ κ°λ¦¬κ³  νΉμ λͺ¨λ₯Ό κ°μΌμλ‘λΆν°μ μ ν κ²½λ‘λ₯Ό μ°¨λ¨νλ κ²μ΄ μ€μνλ€. λ§μ€ν¬λ μ½μ μμ μμ ν κ°λ¦¬λλ‘ μ¬λ°λ₯΄κ² μ°©μ©νμ λ ν¨κ³Όκ° μλ€. νμ§λ§ λμ κ³΅κ³΅μ₯μμμ λͺ¨λ  μ¬λλ€μ λ§μ€ν¬λ₯Ό μ¬λ°λ₯΄κ² μ°©μ©νλμ§ κ²μ¬νλ κ²μ μΆκ°μ μΈ μΈμ μμμ΄ νμνλ€.

λ°λΌμ **μΉ΄λ©λΌλ‘ λΉμΆ°μ§ μ¬λ μΌκ΅΄ μ΄λ―Έμ§**λ§μΌλ‘ μ΄ μ¬λμ΄ **λ§μ€ν¬λ₯Ό μ°κ³  μλμ§, μ°μ§ μμλμ§, μ ννκ² μ΄ κ²μ΄ λ§λμ§** μλμΌλ‘ κ°λ €λΌ μ μλ μμ€νμ΄ νμνλ€.

</br>

# ποΈ λ°μ΄ν°μ

- **λ°μ΄ν°μ κ°μ** : Train 18,900μ₯ / Test 12,600μ₯

- **μ΄λ―Έμ§ ν¬κΈ°** : (384, 512)

- **ν΄λμ€μ λΆλ₯**

  - Mask : Wear, Incorrect, Not Wear
  - Gender : Male, Female
  - Age : <30 (Young), β₯30 and <60 (Middle), β₯60 (Old)
  
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

# π νλ‘μ νΈ κ²°κ³Ό

backboneμ μλ‘ λ€λ₯Έ μ‘°κ±΄μΌλ‘ νμ΅ν μμ 5κ° λͺ¨λΈμ inferenceκ²°κ³Όλ₯Ό hard votingνμ¬ μ μΆν inference κ²°κ³Όκ° κ°μ₯ λμ scoreλ₯Ό λ¬μ±.

- **μμλΈμ μ¬μ©ν λͺ¨λΈ**


  | idx  |                 model                 | f1-score | accuracy |
  | :--: | :-----------------------------------: | :------: | :------: |
  |  1   |         vit_base_patch16_224          |   0.74   |  78.38   |
  |  2   |     swin_base_patch4_window7_224      |   0.71   |  76.47   |
  |  3   |            efficientnet_b4            |   0.71   |  77.88   |
  |  4   |             custom model              |   0.70   |  77.71   |
  |  5   | efficientnet_b4 + RetinaFace + CutMix |   0.70   |  75.92   |

- **μ΅μ’ κ²°κ³Ό**

  f1-score : 0.7436 | accuracy : 79.87
  
  ![Untitled 1](https://user-images.githubusercontent.com/54202082/201047328-3608cbf7-3947-4c15-947d-406ce2c26f9e.png)

</br>

# π¨βπ¨βπ§βπ§ ν κ΅¬μ± λ° μ­ν 

<table>
  <tr height="35px">
    <td align="center" width="180px">
      <a href="https://github.com/iihye">μ΄νμ§_T4177</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/SS-hj">μ΄νμ _T4173</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/brotherhoon-code">κΉνν_T4064</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/iden2t">κΉλμ_T4027</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/ydsong2">μ‘μλ_T4109</a>
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
