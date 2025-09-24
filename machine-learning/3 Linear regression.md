# Linear Regression

앞에서 살펴본 linear classification은 $\pm1$ 로 분류하는 함수이므로 $\mathcal Y=\{+1,-1\}$ 이다. Linear regression의 경우 $\mathcal Y=\mathbb R$ 이다. 함수를 찾는다기 보단 확률 분포를 찾는 과정이다.


## Homoscedasticity
Linear regression을 하기 위한 한 가지 가정은 homoscedasticity(등분산성)이다. 이는 오차항(error term, residual)의 분산이 모든 독립변수 값에 대해 일정하다는 가정이다. 식으로 표현하면 다음과 같다.

$$
\text{Var}(\epsilon_i​∣X_i​)=\sigma^2 \quad \forall i
$$

여기서 오차항은 예측값과 실제값의 차이 $\epsilon_i ​= y_i ​− \hat{y}_i​$ 이다. 이유는 linear regression의 계산 방법인 least squares(최소제곱법)에서 가정을 사용하기 때문이다. 

## Algorithm

Hypothesis가 $h=w^T x$인 linear regression의 계산 알고리즘을 Ordinary Least Squares(OLS)라 한다. Solution은 $E_{in}$을 최소화하는 $w^*$이다.

$$
E_{in}(h) = \frac{1}{N} \sum_{n=1}^N (h(x_n) - y_n)^2
$$

$$
h(x) = w^T x = \sum_{i=0}^d w_i x_i
$$

이를 풀어 쓰면 다음과 같다.

$$
\begin{split}
E_{in}(w) &= \frac{1}{N} \sum_{n=1}^N (w^T x_n - y_n)^2\\
          &= \frac{1}{N} \lVert Xw - y \rVert^2\\
          &= \frac{1}{N} ( w^T X^T X w - 2 w^T x^T y + y^T y )
\end{split}
$$

여기서 $X$는 $x_n$들을 모아놓은 행렬을 뜻한다. 우리의 목표는 $E_{in}$을 최소화 하는 것이다. $E_{in}$이 continuous, differentiable, convex라 하면 gradient가 0인 $w$가 최소이다.

$$
\nabla E_{in}(w) = \frac{2}{N}(X^T X w - X^T y) = 0
$$

따라서 $w$의 조건은 다음과 같다.

$$
X^T X w = X^T y
$$

만약 $X^T X$가 invertible이라면, $w=X^{\dagger} y$로 optimal한 $w$를 구할 수 있다. 여기서 $X^{\dagger}=(X^T X)^{-1} X^T$로 $X$의 pseudo-inverse 이다. $X^T X$가 invertible이 아니더라도 pseudo-inverse가 존재하지만 unique하지는 않다. 결론적으로, $E_{in}$을 최소로 만드는 최적의 $w$를 찾을 수 있다.

## Hat matrix

OLS로 구한 최적의 $w$를 $w_{lin}$라 하면 예측 값은 $\hat y = X w_{lin}$ 이다. 

$$
\hat y = X w_{lin} = X (X^T X)^{-1} X^T y
$$

여기서 $\hat y$와 $y$의 관계를 hat matrix $H$라 한다. 

$$
H = X (X^T X)^{-1} X^T
$$

즉 $\hat y$는 orthogonal projection of $y$ onto column space of $X$ 이다. 이렇게 projection의 역할을 하는 hat matrix를 구해 $\hat y$를 계산하는 방법도 있다. 근본적으로 OLS와 같은 방법이다. 

## Feasibility

Linear regression의 feasibility 또한 확인할 수 있다. Generalization bound의 식은 다음과 같다. 

$$
E_{out}(g) = E_{in}(g) + O \left( \sqrt{ \frac{d}{N} \ln{N}} \right)
$$

Linear regresssion의 경우 여기서 나아가 $E_{in}$과 $E_{out}$을 계산하는 식을 구할 수 있다(Learning From Data, Exercise 3.4). 결과만 이용하자.

$$
E_{in} = 
$$

$$
E_{out} = 
$$

분산 $\sigma^2$과 비교하면 