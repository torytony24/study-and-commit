# Support vector machines

지금까지 linear model에서 상수항이 없는 경우만 다루었다. 이제는 $w^T x + b$ 형태로 함수를 다룬다.

## Linear discriminant function

Linear discriminant function은 다음과 같다.

$$
g(x) = w^T x + b
$$

지금까지는 상수 $b$를 $x$에 포함시켜 같이 보았다. 이제는 상수항 bias도 같이 생각하는 더 일반적인 형태의 모델이다. 전체적인 그림은 다음과 같다.

[[사진]]

$x$와 $w$가 주어질 때, $g(x) = 0$ 이 되는 $H$를 찾는 것이 목표다. Decision surface $H$가 기준이 되어 classify를 하게 되기 때문이다. $H$를 찾으면 되는 이유는 $x$의 $H$로부터의 거리가 $g(x)$가 되기 때문이다. 이는 $g(x)$를 최소화하기만 하면 잘 classify 하는 plane을 찾을 수 있다는 뜻이다.

$$
\begin{split}
g(x) &= w^T x + b \\
     &= w^T \left( x_p + r \frac{w}{\lVert w \rVert} \right) + b \\
     &= (w^T x_p + b) + r \lVert w \rVert \\
     &= r \lVert w \rVert
\end{split}
$$

따라서 $w$의 크기가 1이면 $g(x)$는 $H$로부터의 거리를 의미한다.

## SVM overview

Classify 문제는 여러 가지로 나뉜다. 첫 번째로, linear seperable 한 hard-margin SVM이다. 이는 primal form을 푸는 것과 dual form을 푸는 것, 두 가지로 다시 나뉜다. 두 번째 문제는 linear non-seperable 한 경우다. 이것도 outlier가 있어 almost-linearly seperable 한 soft-margin SVM과 linearly non-seperable한 두 가지로 다시 나뉜다. 후자의 경우 nonlinear transform이나 kernel trick 등을 사용해야 한다.

- Linearly seperable cases (hard-margin SVM)
  - Primal form
  - Dual form
- Non-seperable cases
  - Almost-linearly seperable with outliers (soft-margin SVM)
  - Linearly non-seperable

## Optimization

SVM은 optimization problem의 일종이다. 우리가 풀고자 하는 문제는 제약조건 $g \leq 0$ 또는 $g \geq 0$이 주어질 때, $f$의 최대 또는 최소를 찾는 것이다. 이는 $\mathcal L = f \pm \lambda g$를 이용해 풀어낸다.

$\mathcal L$는 네 가지의 조건을 이용해 푼다.

- $\lambda \geq 0$
- $g \leq 0$ 또는 $g \geq 0$ (문제의 조건)
- $\nabla_x \mathcal{L} = \nabla_x f \pm \lambda \nabla_x g = 0$
- $\lambda g = 0$ (이때, 둘 다 0은 아님)

마지막 조건 $\lambda g = 0$에서 두 가지 케이스로 나뉜다. 먼저 $\lambda = 0$일 경우 $g$는 필요없다. $\mathcal{L} = f$라는 사실을 이용하면 된다. $g=0$일 경우 $\lambda > 0$이 되고, 이 경우 그림처럼 $x$는 $g=0$ 위에 있게 된다.

[[그림]]

