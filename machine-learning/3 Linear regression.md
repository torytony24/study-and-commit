# Linear Regression

앞에서 살펴본 linear classification은 $\pm1$ 로 분류하는 함수이므로 $\mathcal Y=\{+1,-1\}$ 이다. Linear regression의 경우 $\mathcal Y=\mathbb R$ 이다. 함수를 찾는다기 보단 확률 분포를 찾는 과정이다.


## Homoscedasticity
Linear regression을 하기 위한 한 가지 가정은 homoscedasticity(등분산성)이다. 이는 오차항(error term, residual)의 분산이 모든 독립변수 값에 대해 일정하다는 가정이다. 식으로 표현하면 다음과 같다.

$$
\text{Var}(\epsilon_i​∣X_i​)=\sigma^2 \quad \forall i
$$

여기서 오차항은 예측값과 실제값의 차이 $\epsilon_i ​= y_i ​− \hat{y}_i​$ 이다. 이유는 linear regression의 계산 방법인 least squares(최소제곱법)에서 가정을 사용하기 때문이다. 

## Algorithm

Hypothesis가 $h=w^T x$인 linear regression의 계산 알고리즘을 Ordinary Least Squares(OLS)라 한다. OLS의 solution은 $E_{in}$을 최소화하는 $w^*$이다.

$$
E_{in}(h) = \frac{1}{N} \sum_{n=1}^N (h(x_n) - y_n)^2
$$

이때 ㅁㄴㅇㄴㄹㅇ

Solution을 찾는 알고리즘은 다음과 같다.
