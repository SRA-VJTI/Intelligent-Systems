# Vectors
- From the perspective of CS Students, Vectors are ordered lists of numbers.
-  In the case of linear Algebra, a vector will almost always be rooted at the origin. 
- The coordinates of the vector are a pair of numbers that basically give instructions on how to get from the tail of the vector to its tip.
- First no. ( here -2) tells us how far to move in the x-axis and second no. ( here 3) tells how far to move in the y-axis
![300](ML/Linear%20Algebra/Images/img1.png)
- Every pair of numbers gives u only one vector and every vector represents only one pair of number
- In 3d another axis is added. Every triplet of number gives a unique vector in space and vice versa 
![400](ML/Linear%20Algebra/Images/img2.png)

# Vector Addition
![300](ML/Linear%20Algebra/Images/img3.png)

For adding these two vectors move the second one so that its tail sit over the tip of the first one. Now, if you draw a new vector from the tail of the first one towards the tip of the second one that new vector is the sum of the two vectors. 
![300](ML/Linear%20Algebra/Images/img4.png)

![400](ML/Linear%20Algebra/Images/img5.png)

# Scalar Multiplication
- If we multiply a vector by 2 its length becomes twice the original length.
- If we multiply a vector by a negative number then the vector first gets flipped and then stretches.
- This process of stretching, and flipping vectors is known as **scaling** and the numbers which scale vector is known as **scalars**.

# Span
- Unit vector along the x-axis is **i hat**
- Unit vector along the y-axis is **j hat**
- Vector can be represented as a sum of two scaled vectors.
- The x coordinate of a vector is a scalar that scales i hat and the Y coordinate of a vector is a scalar that scales j hat
- ![400](ML/Linear%20Algebra/Images/img6.png)
- i hat and j hat are known as the basis of the coordinate system. This is because this basis vector is actually what the coordinates of  a vector scales
- If we choose a different basis of vector we will get a new coordinate system
- Scaling of two vectors and adding them to represent a vector is known as linear combination of those two vectors.
- Set of all possible vectors that you can reach with a linear combination of a given pair of vectors is called the span of those two vectors.
- The span of most pairs of 2d vectors is all vectors of 2d space except when they are in one line. When they line up their span is just a line
![](ML/Linear%20Algebra/Images/img7.png)
![](ML/Linear%20Algebra/Images/img8.png)

 # Vectors vs Points
 - When dealing with the collection of vectors its common to represent each vector with just a point in space.
 - The point represents the tip of the vector and the tail of the vector is at the origin only.
 - Think of individual vector as arrows 
 - Think of sets of vectors as points
 
# Span of 3d vectors
- similar to 2d vectors
- Take 3 scalars and scale the three vectors and then add them all together. span is set of all possible linear combination
![](ML/Linear%20Algebra/Images/img9.png)
  - If the third vector happens to sit on the span of the first two vectors then the span doesn't change.
  - if third vector is not sitting on the span of the first two, then the span of the three vectors is all possible 3d vectors.
  - Whenever this happens that we can remove a vector without reducing the span,  the relevant terminology is that the vector is linearly dependent.
  - In such a case one of the vectors can be expressed as a linear combination of the other as it is already in the span of the other.
  - If the vector adds another dimension then it is said to be linearly independent.
   ![](ML/Linear%20Algebra/Images/img10.png)
   ![](ML/Linear%20Algebra/Images/img11.png)
   
  
  # **Linear Transformation**
- Transformation is like a function it takes in a vector and gives another vector
- Transformations suggests to think using movements.
-   A transformation is linear if it follows two properties
1. All lines must remain lines
2. Origin must remain fixed
![](ML/Linear%20Algebra/Images/img12.png)
![](ML/Linear%20Algebra/Images/img13.png)
![](ML/Linear%20Algebra/Images/img14.png)
 - In linear transformation, lines remain parallel and evenly spaced.
 - For linear transformation we only need to know where the two basis vectors would land after transformation.
 ![](ML/Linear%20Algebra/Images/img15.png)
 we can deduce where the vector must go based on where i hat and j hat have landed.
 ![](ML/Linear%20Algebra/Images/img16.png)
 -IN general
 ![400](ML/Linear%20Algebra/Images/img17.png)
 - Hence we find where the vector would land after transformation using the above formula.
 - Hence 2D linear transformation is described by just four numbers. The two coordinates where i hat landed and the two coordinates where j has landed
 - It's common to package these four coordinates as 2 * 2 matrix.
![](ML/Linear%20Algebra/Images/img21.png)
To get the transformed vector multiply
the coordinate of the vector with the column of the matrix. 
![](ML/Linear%20Algebra/Images/img20.png)
- In shear transformation, i hat remains the same and only j hat changes.

# Multiplication of matrix as a composition
- If we apply rotation and then shear to a vector, it is the same as applying the transformation of the resultant i hat and j hat to that vector.
- Multiplying matrix has a geometric meaning of applying one transformation and then another.
- First, apply the transformation represented by the matrix on the right then apply the transformation represented by the matrix on the left.
![](ML/Linear%20Algebra/Images/img22.png)

- Let's find the matrix after two transformations:-
1. Let's find the i hat 
 ![](ML/Linear%20Algebra/Images/img23.png)
  2. Let's find the j hat 
  ![](ML/Linear%20Algebra/Images/img24.png)
  - Order of multiplying of matrix matters 
   that is, *M1M2 is not equal to M2M1*
   - Matrix multiplication is associative that is (AB)C=A(BC)
   -  This property is trivial in both cases because in both the cases we are doing first C transformation, then B and then A. Numerically it is difficult to calculate.  
 
   # 3D Transformation 
- the unit vector in the z-direction is k hat
- It is similar to 2d transformation just another axis added. 
- Here transformation is described by 9 numbers.
- These 9 numbers can be packed as a single matrix
- Similarly when two 3d matrices are multiplied first apply the transformation of the right matrix then apply the transformation of the left matrix.
# The Determinant
- When we do linear transformation space is either stretched or squished.
- the factor by which the area of a unit square changes by a linear transformation is its determinant. 
   ![](ML/Linear%20Algebra/Images/img25.png)
   - This factor by which area changes is a determinant. 
   ![](ML/Linear%20Algebra/Images/img26.png)
   - For example, the determinant of a transformation will be 6 if that transformation increases the area by a factor of 6.
   - The determinant would be zero if after transformation span is line only or a single point because the area of any region would be zero.  
   - If the determinant of a matrix is zero, then it will tell that the transformation squishes everything into a smaller dimension.
   - What would negative value in a determinant means:-
   This has to do with the idea of orientation... If orientation is reversed then a negative sign comes and the number still represents the factor by which the area has increased. 
   For Example
   At the start, j hat is to the left of i hat 
    ![](ML/Linear%20Algebra/Images/img27.png)
    After orientation j hat is to the right of i hat, therefore negative sign comes 
     ![](ML/Linear%20Algebra/Images/img28.png)
     - In the case of 3D transformation, the determinant is the volume scaled by a 1v1v1 cube.
     
	![](ML/Linear%20Algebra/Images/img29.png)
	Before transformation
	![](ML/Linear%20Algebra/Images/img30.png)
	After Transformation
	Here the determinant is the volume of the parallelepiped.
	- A determinant with zero value would mean that space has squished into zero volume that is a plane or a line or a point.
	-  Orientation in 3d can be determined by right-hand thumb rule  
			- point forefinger in the direction of i hat 
			- point the middle finger in the direction of j hat
			- now when u point your thumb up it would be in the direction of k hat
	-  If after transformation we can still do that then the orientation is not change and the determinant is positive. And if it only makes sense to do this using the left hand then the orientation is reversed and the determinant is negative. 
	![](ML/Linear%20Algebra/Images/img31.png)

# Inverse Matrix
- Linear Algebra lets us solve certain systems of equations known as linear systems of equations
 ![](ML/Linear%20Algebra/Images/img32.png)
 Thus Ax=v means we are looking for a vector x which after transformation gives the vector v 
 - The solution depends on whether the transformation squishes the space or stretches it that is whether the determinant is zero or not.

**The determinant is not zero**
- In such a case there will always be one and only one vector that lands on v 
- We can find it by doing the transformation in reverse by following where v goes.
- the inverse transformation is the reverse transformation 
For example- 
If A is a counterclockwise rotation by 90 degrees then inverse of A is clockwise rotation by 90 degrees
![](ML/Linear%20Algebra/Images/img33.png)
![](ML/Linear%20Algebra/Images/img34.png)
- Thus if we first apply A and then do A inverse we will reach where we started.
-  Thus A inverse* A = Identity Matrix, that is a matrix that corresponds to nothing. And the transformation that leads to nothing is known as identity transformation 
![400](ML/Linear%20Algebra/Images/img35.png)
- Once we get this inverse matrix multiply it by v and hence we get x.
![400](ML/Linear%20Algebra/Images/img36.png)

**When the determinant is zero**
- That the transformation squishes the space into a smaller dimension
then there is no inverse.
we cannot unsquish a line and turn it into a plane
- It is possible that a solution exists even when the determinant is zero
Only possible when the vector v is on that line only
![400](ML/Linear%20Algebra/Images/img37.png)

# Rank
- When the output of a transformation is a line that is 1d then we say that the transformation has a rank of 1.
- If the vectors land up on the 2d plane after transformation then rank 2.
-  Thus, rank means the number of dimensions in the output of a transformation.
- For example in the case of 2 * 2 matrix rank is generally 2 except when after the transformation space is squished.

#  Column space of a matrix
- The set of all possible output of a matrix whether it's a line or plane or a point is known as the column space of a matrix
- In other words, column space is the span of the column of ur matrix
![400](ML/Linear%20Algebra/Images/img38.png)
- Thus rank is the number of dimensions in the column space of a matrix
- If the rank is equal to the number of columns then we call the matrix full rank.

# Null space
- For full-rank transformation, the only vector that lands on the origin is the zero vector.
- For a matrix that is squished to a smaller dimension, we can have a bunch of vector land at origin.
- This set of vector that land on the origin is known as null space or the kernel of matrix.
- It's the space of all vector that become null.
![400](ML/Linear%20Algebra/Images/img39.png)
![400](ML/Linear%20Algebra/Images/img40.png)
- In the case of a linear system of equations when v happens to be a zero vector then the null space gives you all the possible solutions.

# NonSquare Matrix

- 2d vector input is very different from 3d vector output.
- A 3 * 2 matrix means mapping from 2d to 3d.
- The two columns indicate that the input space has two basis vectors.
- The 3 rows indicate that the landing spot of each of the vectors is described with three coordinates.
- And a 2 * 3 matrix means that we start in a space that has 3 basis vectors and we are landing in a space that has 2 coordinates for each landing spot.
- The transformation from 2d to 1d... 1d is just a number line... Thus, a transformation like this takes in a  2d vector and spits out numbers.

# Dot product
- For the dot product between two vectors v and w imagine projecting w on the line that passes through the origin and the tip of v, multiply the length of this projection by the length of v we get the dot product.
- ![](ML/Linear%20Algebra/Images/img41.png)
 If the projection is towards the opposite direction of v then the dot product is negative ![](ML/Linear%20Algebra/Images/img42.png)
- If the two vectors are perpendicular then the projection of one onto the other is the zero vector then the dot product is 0.
- Here, the order doesn't matter, that is if we project v on w then multiply the length of the projection of v and the length of w, then we get the same dot product.
Numerically dot product is:-
![400](ML/Linear%20Algebra/Images/img43.png)

- if u take a line of evenly spaced dots and apply linear transformation then those dots will be evenly spaced in 1d on a line known as a number line. 
- The basis vector just lands on another number and if we record it we get a 1 * 2  matrix.
- There is some kind of connection between linear transformation that takes vectors to numbers and vectors themselves.
  - Multiplying a 1 * 2 matrix by a 2d vector is same as turning that matrix on its side and taking a dot product
   ![400](ML/Linear%20Algebra/Images/img44.png)
  - This is an example of duality. 
  - duality means natural but surprising correspondence between two types of mathematical things.

# Cross Product
- The cross product of two vectors v and w is the area of the parallelogram they form
- Also, If v is on the right of w then v cross w is positive otherwise negative.
![400](ML/Linear%20Algebra/Images/img45.png)
Here, the cross-product is negative.
- Thus, in the case of cross-product order matters.
![](ML/Linear%20Algebra/Images/img46.png)
- we can find the area by finding the determinant and hence we found the cross product
![500](ML/Linear%20Algebra/Images/img47.png)
![400](ML/Linear%20Algebra/Images/img48.png)
![400](ML/Linear%20Algebra/Images/img49.png)
- technically cross product is not a number it is a vector.
- this new vector length will be the area of the parallelogram and the direction will be perpendicular to the parallelogram.
- In which perpendicular direction is found by right hand rule.
![400](ML/Linear%20Algebra/Images/img50.png)
![400](ML/Linear%20Algebra/Images/img51.png)
- It is easy to remember the process of finding it:-
![400](ML/Linear%20Algebra/Images/img52.png)

# Change of basis
- change of basis matrix is a matrix whose column represents the new basis vector in our language.
- change of basis means having different grid lines but the same origin.
![400](ML/Linear%20Algebra/Images/img53.png)
![](ML/Linear%20Algebra/Images/img54.png)
Therefore a vector is (-1,2) from one basis and (-4,1) from our original basis.
- A matrix whose column represents a new basis vector can be thought of as a transformation that moves our basis vector into new basis vectors.
- but numerically it transforms a vector described in a new basis to our basis.
![400](ML/Linear%20Algebra/Images/img56.png)
![400](ML/Linear%20Algebra/Images/img55.png)

![400](ML/Linear%20Algebra/Images/img57.png)
![400](imgg58.png)
- to find how a vector in our basis looks like in the other basis multiply it by the inverse of the basis matrix.
 ![400](ML/Linear%20Algebra/Images/img59.png)
![400](ML/Linear%20Algebra/Images/img60.png)
# How to translate a matrix
1. start with any vector in a new language
![400](ML/Linear%20Algebra/Images/img61.png)
2. first, convert it into our language using the change of basis matrix this gives the same vector but now in our language.
![400](ML/Linear%20Algebra/Images/img62.png)
3. Apply the transformation matrix to what you get by multiplying it on the left. This tells us where that vector lands after transformation but still in our language.
![400](ML/Linear%20Algebra/Images/img63.png)
4. So in the last step apply the inverse change of the basis matrix to get the transformed vector in the new language.
![400](ML/Linear%20Algebra/Images/img64.png)
- hence this composition of 3 matrices gives the vector in new language after transformation

# Eigenvectors And Eigenvalues


- Most vectors get knocked off their span during the transformation but not all, The vectors which remain on their span even after the transformation are known as Eigenvectors.
- Such vectors are only squished or stretched on their span. The factor they get squished or stretched is known as Eigenvalues.![[ML/Linear Algebra/Images/img65.png]]
![400](ML/Linear%20Algebra/Images/img65.png)
![400](ML/Linear%20Algebra/Images/img66.png)
![400](ML/Linear%20Algebra/Images/img67.png)
- In the case of 3d eigenvector is the axis of rotation and the eigenvalue is 1 because rotation does not squish or stretch.
![400](ML/Linear%20Algebra/Images/IMG68.png)
HERE A*v is matrix vector multiplication and lambda*v is scalar multiplication
- we can write the RHS as
![400](ML/Linear%20Algebra/Images/img69.png)
![400](ML/Linear%20Algebra/Images/img70.png)
- for the above equation to be equal to zero is only possible if it squishes the space.
- for that determinant is zero of the transformation
![400](ML/Linear%20Algebra/Images/img71.png)
- to find if a value lambda is an eigen value subtract it from the transformation matrix and find its determinant.
![400](ML/Linear%20Algebra/Images/img72.png)
- for finding the eigen vector 
![400](ML/Linear%20Algebra/Images/img73.png)
- there are cases when there are no eigenvectors like a rotation by 90 degrees. if we try to find eigenvalue we will get imaginary numbers.
- In case of a shear:-
![400](ML/Linear%20Algebra/Images/img74.png)
- we can also have one eigen value but so many eigenvectors.

# Eigenbasis
- what if both basis vectors are eigenvectors?
- then the matrix would be diagonal and the eigenvalues are there are at the diagonal.
- In the case of a diagonal matrix, diagonal elements are eigenvalues of the basis vector.
- if your coordinate system has a lot of eigenvectors then we can change our coordinate system so that these eigenvectors are the new basis vectors.

# Shortcut for finding eigen value
![400](ML/Linear%20Algebra/Images/img75.png)
for example:-
![400](ML/Linear%20Algebra/Images/img76.png)

# Abstract Vector Spaces
- the function is just another type of vector.
1. like two vectors, we can add two functions.
2. there is a linear transformation in function also, like derivative transforms one function into another function.
- The formal  definition of linearity:- 
![400](ML/Linear%20Algebra/Images/img77.png)
![400](ML/Linear%20Algebra/Images/img78.png)
the idea of grid lines remaining parallel and evenly spaced is just an illustration of what these two properties mean.
- Similarly, in case of function derivative is linear 
 ![400](ML/Linear%20Algebra/Images/img79.png)
 - similar is the case with scaling.
![400](ML/Linear%20Algebra/Images/img80.png)
![400](ML/Linear%20Algebra/Images/img81.png)
![400](ML/Linear%20Algebra/Images/img82.png)
![](ML/Linear%20Algebra/Images/img83.png)
- these are known as Axioms.
- In the modern theory of Linear Algebra, there are 8 Axioms that any vector space must satisfy.


