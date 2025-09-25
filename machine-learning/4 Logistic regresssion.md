# Logistic regression

앞서 살펴본 기계학습 방법의 hypothesis function은 다음과 같다.

- Linear classification: $h(x) = \text{sign}(w^T x)$
- Linear regression: $h(x) = w^T x$

Logistic regression은 다음과 같다. 여기서 $\theta$는 logistic function이다.

- Logistic regression: $h(x) = \theta(w^T) x$

## Logistic function

Logistic function $\theta$는 다음과 같은 함수다. Sigmoid function이라고도 한다.

$$
\theta(s) = \frac{1}{1+e^{-s}}
$$

[??사진??]

