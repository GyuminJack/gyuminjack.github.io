####classification 이후 사용할 수 있는 평가 metric정리


1. precision : P($Y=1$|$\hat{Y}=1$) 
2. recall(sensitivity) : P($\hat{Y}=1$|$Y=1$)
3. specificity(FPR) : P($\hat{Y}=0$|$Y=0$)
4. TPR = 1- FPR = P($\hat{Y}=1$|$Y=0$)

#### ROC / PR(precision-recall)
ROC커브
    - x : 1-specificity(TPR)
    - y : recall(sensitivity)
    - 일반적으로 x가 늘어나면 y가 줄어듬

https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves
The key thing to note is that sensitivity/recall and specificity, which make up the ROC curve, are probabilities conditioned on the true class label. Therefore, they will be the same regardless of what P(Y=1) is.

~~~
roc 커브를 만드는 recall(=sensitivity)와 specificity는 true class의 라벨에 대한 조건부 확률이다.
그러므로 recall과 specificity는 실제 p(y=1)를 신경쓰지 않는다.
~~~

Precision is a probability conditioned on your estimate of the class label and will thus vary if you try your classifier in different populations with different baseline P(Y=1). 
~~~
precision은 만들어진 분류기를 통해 만들어진 라벨에 대한 조건부 확률이며, 
따라서 우리가 P(y=1)이 다른 모집단에서 분류기를 사용하면 값은 달라지게 된다.
- 실제 어느 집단에서는 좋다고 하는 분류기도 다른 모집단으로 가면 p(y=1)에 의해 안좋다고 평가될 수 있다.
- y=1이 작을 수록 precision은 크게 왔다 갔다 하거나, 아예 작은 수로 가거나 할것.
~~~
However, it may be more useful in practice if you only care about one population with known background probability and the "positive" class is much more interesting than the "negative" class. (IIRC precision is popular in the document retrieval field, where this is the case.) This is because it directly answers the question, "What is the probability that this is a real hit given my classifier says it is?".
~~~

하지만, 이 특성은 모집단중 positive의 클래스가 negative보다 더 관심이 있을때 굉장히 유용하게 작용한다. 
(예를 들어 IIRC precision의 경우에는 문서 검색 필드(document retrieval field)에서 많이 사용된다.) 
왜냐하면 precision은 "내 분류기가 실제로 (positive를 positive로)맞춘게 얼마나 되?"라는 질문에 직접적으로 답하기 때문이다.
~~~
Interestingly, by Bayes' theorem you can work out cases where specificity can be very high and precision very low simultaneously. All you have to do is assume P(Y=1) is very close to zero. In practice I've developed several classifiers with this performance characteristic when searching for needles in DNA sequence haystacks.
~~~

흥미롭게도 베이즈 이론을 사용하면, specificity는 높고 precision은 낮은 상황을 발견 할 수 있다. 
P(y=1)을 0에 가깝게 맞춘다면 가능하다. 실제로 나는 몇개의 분류기에서 DNA관련 실험을 신행할때 이런 상황을 본적이 있었다.  
~~~

IMHO when writing a paper you should provide whichever curve answers the question you want answered (or whichever one is more favorable to your method, if you're cynical). If your question is: "How meaningful is a positive result from my classifier given the baseline probabilities of my problem?", use a PR curve. If your question is, "How well can this classifier be expected to perform in general, at a variety of different baseline probabilities?", go with a ROC curve.
~~~
제 생각에는 논문이나 집필의 경우 roc커브와 prcurve 둘다 보여주는것이 좋을 듯 싶다.
(아니면 당신이 cynical하다면 유리한 curve를 보여줘라.)
당신이 "내 분류기의 주어진 baseline probablities에서 실제 양성에 대한 판정 얼마나 의미가 있니?"라고 물어보는 것이라면 PR curve를 사용하고, 
"내 분류기가 여러 환경에서 얼마나 general할 거 같니?" 라고 물어보는 것이라면 ROC커브를 사용해라. 
~~~