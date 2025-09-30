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

Logistic regression도 학습을 위해 error를 정의해야 한다. Linear regression에서 했던 것 처럼 square error로 $E_{in}$을 계산하는 measure 식은 다음과 같다.

$$
E_{in}(h) = \frac{1}{N} \sum^N_{n=1} \left( h(x_n) - \frac{1}{2}(1+y_n) \right)^2
$$

하지만 이 식은 최소화하기 어렵다. 그래서 우리는 convex해서 최소화하기 쉬운 cross-entropy measure를 이용한다.

$$
E_{in}(h) = \frac{1}{N} \sum^N_{n=1} \ln{\left( 1 + e^{-y_n w^T x_n} \right)}
$$

이 error measure를 유도하는 방법은 두 가지가 있다.

## Likelyhood

첫 번째 유도 방법은 likelyhood를 이용하는 것이다. 데이터가 예측값과 일치할 확률은 앞서 구했다. 이 확률이 likelyhood이다. 

$$
P(y|x) = \theta(y w^T x)
$$

Error를 minimize하는 방향으로, 즉 모든 데이터가 예측값과 일치할 확률을 계산해서 이를 maximize하면 된다. 

$$
P(y_1|x_1)P(y_2|x_2)\cdots P(y_N|x_N) = \prod^N_{n=1}P(y_n|x_n)
$$

$-$를 붙여 minimize 하는 식으로 만들고, 계산의 편의를 위해 $\ln$을 붙이고 $N$으로 나누면 다음과 같다.



