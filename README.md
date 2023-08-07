# Siamese Network for few-shot learning for image recognition

## Load Dataset
- 資料集架構
1. 共用40個類別(40個人臉)：s1~s40
![image](https://raw.githubusercontent.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/7a73852d3439f11b84fd1b8d0c79be83e1ae0046/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/Images/1.png)
2. 每一個類別10張圖片
![image](https://raw.githubusercontent.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/7a73852d3439f11b84fd1b8d0c79be83e1ae0046/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/Images/3.png)
- show image
1. 測試輸入的圖片對(sample,20000對)
2. 輸入圖片對的labels(是來自同類還是來自不同類)
3. 網路的預測輸出(相似還是不相似)
- 資料集分割
1. 75%分給訓練(15000對)，25%分給測試(5000對)
## Data Loader
- get_data
1. x_geuine_pair、x_imposite_pair，抓取真圖片對(兩圖片來自相同類，即同一人)與假圖片對(兩圖片來自不同類，即不同人)
2. y_genuine、y_imposite，給圖片對設置的labels(即若圖片對為真設為0，為假則設為1)  
x_geuine_pair np.zeros([total_sample_size, 2, dim1, dim2, 1],dtype=('float32'))  
第一個元素為總sample個數；第二個元素為有幾個圖片，這裡使用有兩張圖片組成；第三個元素為輸入圖片的色彩通道，因為為黑白圖片所以只有一個通道；第四與第五個元素為圖片的寬與高
## Training
- Energy Fumction：L2 distance
- Loss function：contrastive_loss
  -公式#1：
   ![image](https://i.stack.imgur.com/zDtA0.png)  
   來自同類別的圖片(相似)設為1，來自不同類別(不相似)的圖片設為0，  
  -公式#2：
  ![image](https://pic3.zhimg.com/80/v2-bfa48776c69d7e2cbfcf9bc118e5e86e_720w.webp)  
  最初在給圖片對做label的時候，要將來自同類別的圖片(相似)設為0，來自不同類別的圖片設為1，  
  原作者使用的是#1公式，這裡使用#2公式。  
- Optimizer：RMS
## Result
- Compute Accuracy
  -contrastive_loss使用#1 的公式：  
  因為來自同類別的圖片(相似)設為1，來自不同類別(不相似)的圖片設為0，輸出(pred)為輸入圖片對的特徵向量距離，  
  1) predictions.ravel()  
  2) predictions.ravel() < 0.5 條件判斷，輸出boolean矩陣  
  3) labels[predictions.ravel() < 0.5]：使用boolean矩陣作為索引，當索引為True，返回labels對應索引處的元素，即返回神經網路認為(預測)為同類別的圖片。   
  4) 有可能predictions認為是True(特徵距離小，來自同類圖片)，但實際上labels為0(來自不同類圖片)，因為這裡設定來自同類別的圖片為1，因此可以直接加總labels[predictions.ravel() < 0.5]的值並做mean來得到準確值  
  -contrastive_loss使用 #2 公式：  
則來自同類別的圖片(相似)設為0，來自不同類別的圖片設為1，  
而predictions認為是True(特徵距離小，來自同類圖片)，如果預測正確的話，labels為0，所以加總labels[predictions.ravel() < 0.5]的值越小代表預測的越準確，  
因此加總labels[predictions.ravel() < 0.5]，實際上是在計算"1"的個數，也就是在計算預測的錯誤率。  
因此這裡如果要算預測準確率的話，使用1-compute_accuracy(pred, y_test)(錯誤率) 來得到。  
