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

이 함수는 뒤에서 사용할 중요한 성질을 가진다.

$$
1 - \theta(s) = \theta(-s)
$$

## Big picture

[??사진??]

Logistic regression의 큰 그림을 보자. 입력 데이터 $x$에 대해 먼저 linear하게 $s=w^Tx$를 계산하면, $s$를 logistic function에 대입해 0과 1로 smooth하게 분류한다. 이는 곧 확률로 볼 수 있다.

$$
\mathbb{P}[y=+1|x] = h(x) = \theta(w^Tx)
$$

반대로 $y=-1$일 확률은 1에서 뺀 값이다.

$$
h(-x) = \theta(-w^Tx) = 1 - \theta(w^Tx) = 1 - h(x)
$$

이를 합쳐서 쓰면 다음과 같다.

$$
\begin{split}
P(y|x) &= h(x)^{\llbracket y=+1 \rrbracket} + (1-h(x))^{\llbracket y=-1 \rrbracket}\\
       &= h(x)^{\llbracket y=+1 \rrbracket} + h(-x)^{\llbracket y=-1 \rrbracket}\\
       &= h(yx)\\
       &= \theta(yw^Tx)
\end{split}
$$

## Error measure

Logistic regression도 학습을 위해 error를 정의해야 한다. Linear regression에서 했던 것 처럼 square