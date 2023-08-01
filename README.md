# Siamese Network for few-shot learning for image recognition

## Load Dataset
- 資料集架構
1. 共用40個類別(40個人臉)：s1~s40
![image](https://raw.githubusercontent.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/7a73852d3439f11b84fd1b8d0c79be83e1ae0046/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/Images/1.png)
2. 每一個類別10張圖片
![image](https://raw.githubusercontent.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/7a73852d3439f11b84fd1b8d0c79be83e1ae0046/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/Images/3.png)
- show image
1. 測試輸入的圖片對
2. 輸入圖片對的labels(是來自同類還是來自不同類)
3. 網路的預測輸出(相似還是不相似)

## Data Loader
