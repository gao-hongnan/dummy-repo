---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# A First Look at Dot Products

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Sequence, TypeVar, Optional
import matplotlib.pyplot as plt
import numpy as np
import rich

import sys
from pathlib import Path

def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.visualization.style import use_svg_display
    from omnivault.linear_algebra.plotter import (
        VectorPlotter2D,
        VectorPlotter3D,
        add_vectors_to_plotter,
        add_text_annotations,
    )
    from omnivault.linear_algebra.vector import Vector2D, Vector3D
else:
    raise ImportError("Root directory not found.")

use_svg_display()
```

In this introductory section, we'll take our first look at the concept of the
dot product, a fundamental operation in linear algebra with far-reaching
implications in various fields, particularly in machine learning and analytical
geometry. We'll revisit and explore this concept more rigorously later in the
series, especially after we've established a solid understanding of vector
spaces.

The dot product, also known as the scalar product, is a way to multiply two
vectors that results in a scalar (a single number). This operation is key to
understanding many aspects of vectors and their interactions, especially in the
context of machine learning, where it's used in tasks ranging from projections
and similarity measurements to more complex operations in algorithms and data
transformations.

In machine/deep learning applications, such as
**[neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network)**
and
**[support vector machines](https://en.wikipedia.org/wiki/Support-vector_machine)**,
the dot product serves as a building block for understanding how data points
relate to each other in feature space. It's also a stepping stone towards more
advanced concepts in analytical geometry, where the dot product plays a crucial
role in defining angles and distances between vectors.

## Vector Multiplications

This section introduces one of the most important idea in Linear Algebra, the
**Dot Product**. Since
[Wikipedia](https://en.wikipedia.org/wiki/Dot_product)[^dot_product] has a
wholesome introduction, we will be copying over some definitions from it.

### Dot Product

#### Algebraic Definition (Dot Product)

The dot product of two vectors
$\color{red}{\mathbf{a} =  \begin{bmatrix} a_1  \; a_2  \; \dots \; a_n \end{bmatrix}^{\rm T}}$
and
$\color{blue}{\mathbf{b} =  \begin{bmatrix} b_1 & b_2  & \dots & b_n \end{bmatrix}^{\rm T}}$
is defined as:

$$\mathbf{\color{red}\mathbf{a}}\cdot\mathbf{\color{blue}\mathbf{b}}=\sum_{i=1}^n {\color{red}a}_i{\color{blue}b}_i={\color{red}a}_1{\color{blue}b}_1+{\color{red}a}_2{\color{blue}b}_2+\cdots+{\color{red}a}_n{\color{blue}b}_n$$

where $\sum$ denotes summation and $n$ is the dimension of the vector space.
Since **vector spaces** have not been introduced, we just think of it as the
$\mathbb{R}^n$ dimensional space.

##### Example of Dot Product

For instance, in 3-dimensional space, the **dot product** of column vectors
$\begin{bmatrix}1 & 3 & -5\end{bmatrix}^{\rm T}$ and
$\begin{bmatrix}4 & -2 & -2\end{bmatrix}^{\rm T}$

$$
\begin{align}
\ [{\color{red}1, 3, -5}] \cdot  [{\color{blue}4, -2, -1}] &= ({\color{red}1} \times {\color{blue}4}) + ({\color{red}3}\times{\color{blue}-2}) + ({\color{red}-5}\times{\color{blue}-1}) \\
&= 4 - 6 + 5 \\
&= 3
\end{align}
$$

---

> "Vector as Matrices"

We are a little ahead in terms of the definition of Matrices, but for people
familiar with it, or have worked with `numpy` before, we know that we can
interpret a row vector of dimension $n$ as a matrix of dimension $1 \times n$.
Similarly, we can interpret a column vector of dimension $n$ as a matrix of
dimension $n \times 1$. With this interpretation, we can perform a so called
"matrix multiplication" of the row vector and column vector. The result is the
dot product. We will go in details when we get to it.

If vectors are treated like row matrices, the dot product can also be written as
a matrix multiplication.

$$\mathbf{\color{red}a} \cdot \mathbf{\color{blue}b} = \mathbf{\color{red}a}^\mathsf T \mathbf{\color{blue}b}$$

Expressing the above example in this way, a 1 × 3 matrix **row vector** is
multiplied by a 3 × 1 matrix **column vector** to get a 1 × 1 matrix that is
identified with its unique entry:

$$
    \begin{bmatrix}
    \color{red}1 & \color{red}3 & \color{red}-5
    \end{bmatrix}
    \begin{bmatrix}
    \color{blue}4 \\ \color{blue}-2 \\ \color{blue}-1
    \end{bmatrix} = \color{purple}3
$$

#### Dot Product (Geometric definition)

In [Euclidean space](Euclidean_space "wikilink"), a
[Euclidean vector](Euclidean_vector "wikilink") is a geometric object that
possesses both a magnitude and a direction. A vector can be pictured as an
arrow. Its magnitude is its length, and its direction is the direction to which
the arrow points. The magnitude of a vector **a** is denoted by
$\left\| \mathbf{a} \right\|$. The dot product of two Euclidean vectors **a**
and **b** is defined by

$$\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta ,$$

where $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$.

<img src="https://storage.googleapis.com/reighns/reighns_ml_projects/docs/linear_algebra/linear_algebra_theory_intuition_code_chap3_fig_3.1_scalar_projection_and_dot_product.PNG" style="margin-left:auto; margin-right:auto"/>
<p style="text-align: center">
    <b>Fig 3.11: Diagram of Scalar Projection and Dot Product; By Hongnan G.</b>
</p>

##### Scalar projections

TODO: To motivate the geometric interpretation, we should see the example on
scalar projections.

##### Sign of the DOT Product is determined by the Angle in between the two vectors

The geometric definition can be re-written as follows:

$$
\begin{equation} \label{eq1}
\begin{split}
\mathbf{a}\cdot\mathbf{b} &=\|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta \implies \cos(\theta) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} \implies \theta = \cos^{-1}\left(\frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}\right)
\end{split}
\end{equation}
$$

which essentially means that one can find the angle between two known vectors in
any dimensional space.

Mike X Cohen explains in **Linear Algebra: Theory, Intuition, Code, 2021. (pp.
51-52)** how the **sign** of the dot product is determined solely by the angle
between the two vectors. **By definition**,
$\mathbf{a
}\cdot\mathbf{b} = \|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta$, we know
that the sign (positive or negative) of the dot product
$\mathbf{a} \cdot \mathbf{b}$ is solely determined by $\cos \theta$ since
$\|\mathbf{a}\| \|\mathbf{b}\|$ is always positive.

-   **Case 1 ($0< \theta < 90$): This implies that
    $\cos \theta > 0 \implies \|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta > 0 \implies \mathbf{a}\cdot\mathbf{b} > 0$.**
-   **Case 2 ($90 < \theta < 180$): This implies that
    $\cos \theta < 0 \implies \|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta < 0 \implies \mathbf{a}\cdot\mathbf{b} < 0$.**
-   **Case 3 ($\theta = 90$): This is an important property, for now, we just
    need to know that since $\cos \theta = 0$, then
    $\mathbf{a} \cdot \mathbf{b} = \mathbf{0}$. These two vectors are
    orthogonal.**
-   **Case 4 ($\theta = 0$ or $\theta = 180$): This implies that
    $\cos \theta = 1 \implies \|\mathbf{a}\|\ \|\mathbf{b}\|\cos\theta = \|\mathbf{a}\|\ \|\mathbf{b}\|$.
    We say these two vectors are collinear.**

> "Consequence of Case 4"

A simple consequence of case 4 is that if a vector $\mathbf{a}$ dot product with
itself, then by case 4, we have
$\mathbf{a} \cdot \mathbf{a} = \|\mathbf{a}\|^2 \implies \|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}}$
which is the formula of the [Euclidean length](Euclidean_length "wikilink") of
the vector.

<img src="https://storage.googleapis.com/reighns/reighns_ml_projects/docs/linear_algebra/linear_algebra_theory_intuition_code_chap3_fig_3.2.svg" style="margin-left:auto; margin-right:auto"/>
<p style="text-align: center">
    <b>Fig 3.2: Sign of Dot Product and Angle between two vectors; By Hongnan G.</b>
</p>

#### Intuition

-   https://flexbooks.ck12.org/cbook/ck-12-college-precalculus/section/9.6/primary/lesson/scalar-and-vector-projections-c-precalc/
-   https://www.quora.com/What-are-the-geometrical-meanings-of-a-dot-product-and-cross-product-of-a-vector
-   https://math.stackexchange.com/questions/805954/what-does-the-dot-product-of-two-vectors-represent

#### Properties of Dot Product

The **dot product**[^dot_product] fulfills the following properties if **a**,
**b**, and **c** are real [vectors](<vector_(geometry)> "wikilink") and
$\lambda$ is a [scalar](<scalar_(mathematics)> "wikilink").

1.  **[Commutative](Commutative "wikilink"):**

    $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a} ,$ which follows
    from the definition (_θ_ is the angle between **a** and **b**):
    $\mathbf{a} \cdot \mathbf{b} = \left\| \mathbf{a} \right\| \left\| \mathbf{b} \right\| \cos \theta = \left\| \mathbf{b} \right\| \left\| \mathbf{a} \right\| \cos \theta = \mathbf{b} \cdot \mathbf{a} .$

2.  **[Distributive](Distributive_property "wikilink") over vector addition:**

    $\mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c} .$

3.  **[Bilinear](bilinear_form "wikilink")**:

    $\mathbf{a} \cdot ( \lambda \mathbf{b} + \mathbf{c} ) = \lambda ( \mathbf{a} \cdot \mathbf{b} ) + ( \mathbf{a} \cdot \mathbf{c} ) .$

4.  **[Scalar multiplication](Scalar_multiplication "wikilink"):**

    $( \lambda_1 \mathbf{a} ) \cdot ( \lambda_2 \mathbf{b} ) = \lambda_1 \lambda_2 ( \mathbf{a} \cdot \mathbf{b} ) .$

5.  **Not [associative](associative "wikilink")**:

    This is because the dot product between a scalar (**a ⋅ b**) and a vector
    (**c**) is not defined, which means that the expressions involved in the
    associative property, (**a ⋅ b**) ⋅ **c** or **a** ⋅ (**b ⋅ c**) are both
    ill-defined. Note however that the previously mentioned scalar
    multiplication property is sometimes called the "associative law for scalar
    and dot product" or one can say that "the dot product is associative with
    respect to scalar multiplication" because
    $\lambda (\mathbf{a} \cdot \mathbf{b}) = (\lambda \mathbf{a}) \cdot \mathbf{b} = \mathbf{a} \cdot (\lambda
    \mathbf{b})$.

6.  **[Orthogonal](Orthogonal "wikilink"):**

    Two non-zero vectors **a** and **b** are _orthogonal_ if and only if
    $\mathbf{a} \cdot \mathbf{b} = \mathbf{0}$.

7.  **No [cancellation](cancellation_law "wikilink"):**

    Unlike multiplication of ordinary numbers, where if $ab=ac$ then _b_ always
    equals _c_ unless _a_ is zero, the dot product does not obey the
    [cancellation law](cancellation_law "wikilink").

8.  **[Product Rule](Product_Rule "wikilink"):**

        If **a** and **b** are (vector-valued) [differentiable functions](differentiable_function "wikilink"), then the derivative, denoted by a prime ' of $\mathbf{a} \cdot \mathbf{b}$ is given by the rule $(\mathbf{a} \cdot \mathbf{b})' = \mathbf{a}' \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b}'$.

#### Application to the law of cosines

A triangle with lines a, b and c is presented in figure 3.31, a and b are
separated by angle _θ_, then the **law of cosine** states that

$$\Vert c \Vert^2 = \Vert a \Vert^2 + \Vert b \Vert^2 - 2ab\cos(\theta)$$

##### Proof

The proof is from Wikipedia[^law_of_cosine_vector_proof].

Denote

$$\overrightarrow{CB}=\vec{a}, \ \overrightarrow{CA}=\vec{b}, \ \overrightarrow{AB}=\vec{c}$$

Therefore,

$$\vec{c} = \vec{a}-\vec{b}$$

Taking the dot product of each side with itself:

$$\vec{c}\cdot\vec{c} = (\vec{a}-\vec{b})\cdot(\vec{a}-\vec{b})$$
$$\Vert\vec{c}\Vert^2 = \Vert\vec{a}\Vert^2 + \Vert\vec{b}\Vert^2 - 2\,\vec{a}\cdot\vec{b}$$

Using the identity (see [[Dot product]])

$$\vec{u}\cdot\vec{v} = \Vert\vec{u}\Vert\,\Vert\vec{v}\Vert \cos\angle(\vec{u}, \ \vec{v})$$

leads to

$$\Vert\vec{c}\Vert^2 = \Vert\vec{a}\Vert^2 + {\Vert\vec{b}\Vert}^2 - 2\,\Vert\vec{a}\Vert\!\;\Vert\vec{b}\Vert \cos\angle(\vec{a}, \ \vec{b})$$

The result follows.

---

In short, the proof is presented below:

$$
\begin{align}
\mathbf{\color{orange}c} \cdot \mathbf{\color{orange}c}  & = ( \mathbf{\color{red}a} - \mathbf{\color{blue}b}) \cdot ( \mathbf{\color{red}a} - \mathbf{\color{blue}b} ) \\
 & = \mathbf{\color{red}a} \cdot \mathbf{\color{red}a} - \mathbf{\color{red}a} \cdot \mathbf{\color{blue}b} - \mathbf{\color{blue}b} \cdot \mathbf{\color{red}a} + \mathbf{\color{blue}b} \cdot \mathbf{\color{blue}b} \\
 & = \mathbf{\color{red}a}^2 - \mathbf{\color{red}a} \cdot \mathbf{\color{blue}b} - \mathbf{\color{red}a} \cdot \mathbf{\color{blue}b} + \mathbf{\color{blue}b}^2 \\
 & = \mathbf{\color{red}a}^2 - 2 \mathbf{\color{red}a} \cdot \mathbf{\color{blue}b} + \mathbf{\color{blue}b}^2 \\
\mathbf{\color{orange}c}^2 & = \mathbf{\color{red}a}^2 + \mathbf{\color{blue}b}^2 - 2 \mathbf{\color{red}a} \mathbf{\color{blue}b} \cos \mathbf{\color{purple}\theta} \\
\end{align}
$$

which is the [law of cosines](law_of_cosines "wikilink").

<img src="https://storage.googleapis.com/reighns/reighns_ml_projects/docs/linear_algebra/linear_algebra_theory_intuition_code_chap3_fig_3.31_law_of_cosine.svg" style="margin-left:auto;margin-center:auto; margin-right:auto"/>
<p style="text-align: center">
    <b>Fig 3.31; Law of Cosine;</b>
</p>

### Cauchy-Schwarz Inequality

#### Definition (Cauchy-Schwarz Inequality)

Let two vectors $\mathbf{v}$ and $\mathbf{w}$ be in a field $\F^n$, then the
inequality

$$|\mathbf{v}^\top \mathbf{w}| \leq \Vert \mathbf{v} \Vert \Vert \mathbf{w} \Vert$$

holds.

---

> This inequality provides an **upper bound** for the dot product between two
> vectors; in other words, the absolute value of the dot product between two
> vectors cannot be larger than the product of the norms of the individual
> vectors. Note that the inequality can become an equality if and only if both
> vectors are the zero vector $\mathbf{0}$ or if one vector (either one) is
> scaled by the other vector $\mathbf{v} = \lambda \mathbf{w}$. - **Mike X
> Cohen, Linear Algebra: Theory, Intuition, Code**

If you wonder why when $\mathbf{v} = \lambda \mathbf{w}$ implies equality, it is
apparent if you do a substitution as such

$$|\mathbf{v}^\top \mathbf{w}| = |\lambda \mathbf{w}^\top \mathbf{w}| = \lambda |\mathbf{w}^\top \mathbf{w}| = \lambda \|\mathbf{w}\|^2 = \lambda \|\mathbf{w}\| \|\mathbf{w}\| = \|\mathbf{v}\| \|\mathbf{w}\|$$

where we used the fact that $\mathbf{w}^\top \mathbf{w} = \|\mathbf{w}\|^2$ by
definition.

The author decided to include this inequality here because this theorem is
always used in many proofs. He then shows a use case in the Geometric
Interpretation of the Dot Product.

#### Proof of Algebraic and Geometric Equivalence of DOT Product

Read
[here](https://proofwiki.org/wiki/Equivalence_of_Definitions_of_Dot_Product) and
also page 54-56 of Mike's book.

[^dot_product]: https://en.wikipedia.org/wiki/Dot_product
