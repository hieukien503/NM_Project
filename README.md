<h1>Thuật toán QR trong việc tìm trị riêng và vector riêng của ma trận</h1>
<h2>1. Mở đầu</h2>
<p>Trong lĩnh vực đại số tuyến tính và các ứng dụng của nó, việc tìm trị riêng (eigenvalue) và vector riêng (eigenvector) của ma trận đóng vai trò vô cùng quan trọng. Đây là những công cụ cơ bản để phân tích và xử lý nhiều vấn đề trong khoa học máy tính, vật lý, kinh tế và nhiều ngành khoa học khác. Các trị riêng và vector riêng không chỉ cung cấp thông tin về cấu trúc của ma trận mà còn giúp giải quyết các bài toán liên quan đến biến đổi ma trận, phân tích phổ, và nhiều ứng dụng thực tiễn như nén ảnh, nhận dạng mẫu, và phân tích dữ liệu.

Trong số nhiều phương pháp tìm trị riêng và vector riêng, thuật toán QR nổi bật nhờ tính hiệu quả và độ ổn định số học cao. Thuật toán này dựa trên việc phân rã QR của ma trận và được cải tiến qua nhiều phiên bản nhằm tối ưu tốc độ hội tụ cũng như khả năng xử lý ma trận lớn. Nhờ vậy, thuật toán QR trở thành một trong những công cụ mạnh mẽ nhất trong việc giải các bài toán trị riêng của ma trận.

Đề tài này tập trung nghiên cứu chi tiết thuật toán QR, từ cơ sở lý thuyết đến các bước triển khai thực tế. Bên cạnh đó, chúng tôi cũng sẽ phân tích độ phức tạp, ưu điểm, nhược điểm của thuật toán, cũng như các cải tiến quan trọng giúp tăng cường hiệu quả trong thực hành. Việc nắm vững thuật toán QR không chỉ giúp hiểu sâu hơn về các phương pháp xử lý ma trận mà còn mở ra nhiều ứng dụng quan trọng trong các lĩnh vực kỹ thuật và khoa học tính toán.

</p>
<h2>2. Ý tưởng của bài toán</h2>
<p>Gọi $A$ là ma trận thực, vuông cấp $n$ mà chúng ta muốn tìm trị riêng và vector riêng. Với cách truyền thống, chúng ta sẽ cần phải giải phương trình $det(A - \lambda I) = 0$ để tìm $\lambda$, rồi từ $\lambda$ để tìm ra vector riêng. Tuy nhiên khi ma trận $A$ ngày càng lớn ($n$ càng lớn), việc tính $det(A - \lambda I)$ sẽ rất khó khăn, không kể đến việc phải phương trình đa thức bậc cao (vốn không có công thức nghiệm tổng quát với đa thức có bậc lớn hơn $4$).

Do đó, chúng ta có thể thử phân rã A thành tích của hai ma trận $Q$ và $R$, với $Q$ là ma trận trực giao (là ma trận thỏa mãn $Q^{-1}=Q^T$ và các vector hàng và cột trong $Q$ là các vector trực chuẩn), và $R$ là ma trận tam giác trên. Sau khi phân rã $A$ thành hai ma trận $Q$ và $R$ như trên, chúng ta sẽ cập nhật $A = RQ$ và lặp lại các bước trên, cho đến khi $A$ hội tụ về dạng ma trận tam giác (tạm gọi là $A_{conv}$), và các phần tử nằm trên đường chéo của $A_{conv}$ chính là trị riêng của ma trận $A$ ban đầu. Với mỗi $1\le i \le n$, trị riêng thứ $i$ của ma trận $A_{conv}$ có vector riêng là cột thứ $i$ của $Q$.

</p>
<h2>3. Phân rã QR</h2>
<p>Phân rã QR là một phần quan trọng, giúp chúng ta xây dựng chuỗi ma trận $A$ của thuật toán QR. Ý tưởng của phương pháp là ta sẽ phân rã từng cột của ma trận $A$ thành tổ hợp tuyến tính của các vector cột trực chuẩn của $Q$. Có $3$ phương pháp được sử dụng:
  <ul>
    <li>Gram-Schmidt Process</li>
    <li>Householder Reflections</li>
    <li>Givens Rotations</li>
  </ul>
</p>
<h3>3.1. Quy trình Gram-Schmidt</h3>
<p>Quy trình Gram-Schmidt là quy trình dùng để xây dựng một tập hợp gồm $k$ vector ($k\ge 2$) đôi một vuông góc với nhau.

Ta định nghĩa tích vô hướng của $2$ vector $u$ và $v$ như sau: $\langle u,v\rangle=u^Tv$, phép chiếu vuông góc của vector $a$ lên vector $u$, kí hiệu $proj_{u} a$ được tính bằng công thức:

```math
proj_{u} a = \dfrac{\langle u,a\rangle}{\langle u,u\rangle}u
```
Khi đó, với $k$ vector (khác vector $0$) độc lập tuyến tính $v_1,v_2,v_3,\cdots,v_k$, quy trình Gram-Schmidt tạo ra một bộ $k$ vector $u_1,u_2,u_3,\cdots,u_k$ như sau:
```math
u_1 = v_1,~e_1 = \dfrac{u_1}{\left|\left|u_1\right|\right|}
```
```math
u_2 = v_2 - proj_{u_1} v_2,~e_2 = \dfrac{u_2}{\left|\left|u_2\right|\right|}
```
```math
u_3 = v_3 - proj_{u_1} v_3 - proj_{u_2} v_3,~e_3 = \dfrac{u_3}{\left|\left|u_3\right|\right|}
```
```math
\vdots
```
```math
u_k = v_k - \sum_{i=1}^{k-1} proj_{u_i} v_k,~e_k = \dfrac{u_k}{\left|\left|u_k\right|\right|}
```
Áp dụng quy trình Gram-Schmidt cho ma trận $A$, với chú ý rằng nếu $rank(A) = k < n$, khi đó chỉ $k$ cột đầu tiên của $Q$ là $k$ vector trực chuẩn cơ sở của không gian con sinh bởi các cột của ma trận $A$.
Ta thu được $Q=\left[e_1\quad e_2 \quad\cdots\quad e_n\right]$ và
```math
R=\begin{bmatrix}
  \langle e_1,a_1\rangle&\langle e_1,a_2\rangle&\langle e_1,a_3\rangle&\cdots&\langle e_1,a_n\rangle\\
  0&\langle e_2,a_2\rangle&\langle e_2,a_3\rangle&\cdots&\langle e_2,a_n\rangle\\
  0&0&\langle e_2,a_3\rangle&\cdots&\langle e_2,a_n\rangle\\
  \vdots&\vdots&\vdots&\ddots&\vdots\\
  0&0&0&\cdots&\langle e_n,a_n\rangle
\end{bmatrix}
```
Tuy nhiên, cách này có một nhược điểm là không ổn định về mặt số học để tính toán. Lí do là ma trận $A$ của chúng ta là ma trận thực, và các vector $u_k$ sinh ra đều có một sai số nào đó do làm tròn số, và sai số này được sử dụng cho bước tính $u_{k+1}$.

Do đó, chúng ta cần cải tiến phương pháp này, và gọi nó là phương pháp Modified Gram-Schmidt.
</p>
<h3>3.2. Modified Gram-Schmidt Process</h3>
<p>Phương pháp này ra đời với mục đích: Tại bước thứ $k + 1$, loại bỏ các sai số của $u_i,~1\le i \le k$ ra khỏi việc tính $u_{k+1}$, $u_{k+2}$,.... Muốn làm được việc này, tại bước thứ $k + 1$, ta chỉ cần loại bỏ $proj_{u_i} a_k,~1\le i \le k$ ra khỏi công thức tính $u_{k+1}$ là xong.

Công thức mới sẽ là:
```math
q_i = \dfrac{a_i}{\left|\left|a_i\right|\right|},~a_j = a_j - proj_{q_j} a_i,~1 \le j \le n,~j + 1 \le i \le n
```
Khi đó $Q=\left[q_1\quad q_2\quad\cdots\quad q_n\right]$
</p>
<h3>3.3. Householder Reflections</h3>
<p>Phép đối xứng Householder (hay phép biến đổi Householder) là phép biến đổi vector, trong đó ta cần biến đổi đối xứng vector $a$ cho trước thành vector $a'$ có giá trùng với giá của vector $e$ nào đó mà bảo toàn được độ lớn của vector $a$. Ở đây, ta chọn $e_1 = [1\quad 0\quad 0\quad\cdots\quad 0]$

Để tìm vector $a'$, ta cần tìm được vector $u$ sao cho phép đối xứng qua vector $u$ biến $a$ thành $a'$, như hình:
![1_JgR_uSU3-8dGNV-3nBnOPg.png](<https://media-hosting.imagekit.io/def17bc41fae4ff0/1_JgR_uSU3-8dGNV-3nBnOPg.png?Expires=1838060607&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=LVzQmikekmM-UtK~~7bBBm47YoICQ-30lBJnu2EiVUTF61yQHJPy1cgN2MomkNLVBWvJSJ107eJf3lgo3dK5tIjMo0cVLjiHEvNe2IJ183UgmZlsGTMb~OKLDCo4GLr8xwe~9MUK9cL9Pd3qkBgL-uwU1lPThr2PQFILzGEU11vXYKJSeRjz0N32oIXQPmeBUga-7RIeJoPecdpAfuSSDrSUkhbnavAVwwVahwsz3xEwNcegynWUinJPZJ4l-DxLxwPHMpLmIA3tzDVrTS4fc5uo0-aadl2PsWdhNDTsLeT7elNubdwWyNckMWrObloiYXSB1W2yPiFwli9vNAAd-g__>)

Bằng phép biến đổi đơn giản, ta thu được các công thức sau cho cột đầu tiên của ma trận $A$:
```math
u = x + sign(a_{1})\left|\left|a_1\right|\right|e_1
```
```math
v = \dfrac{u}{\left|\left|u\right|\right|}
```
Ma trận Householder được xác định bởi công thức:
```math
H=I-2vv^T
```
Khi đó, ta có $A_1 = HA, Q = QH^T$.

Tuy nhiên, đó chỉ mới là cột đầu tiên của ma trận $A$, để phân rã các cột còn lại, ta thực hiện theo nguyên tắc: Tại bước thứ $k + 1,1\le k \le n - 2$, ta thực hiện phân rã cột thứ $k + 1$ của ma trận $A'_k$, với $A'_k$ là ma trận con thu được bằng cách loại bỏ đi hàng thứ $k$ và cột thứ $k$ của ma trận $A_k$. Khi đó, ma trận Householder của chúng ta cần phải được điều chỉnh lại sau mỗi bước (để thỏa điều kiện nhân ma trận, bằng cách mở rộng ma trận $H_k$ thành:
```math
H_k = \begin{bmatrix}
  I_{k - 1} & 0_{k - 1} \\
  0_{k - 1} & H_k
\end{bmatrix}
```
</p>
<h3>3.4. Givens Rotations</h3>
<p>Khác với phép biến đổi Householder, phép biến đổi Givens thực hiện biến đổi vector cột của ma trận $A$ bằng phép quay một góc $\theta$ nào đó.

Trong không gian 2 chiều ($\mathbb{R}_2$), ta định nghĩa ma trận quay Givens $G_2$ như sau:
```math
G_2=\begin{bmatrix}
  \cos{\theta} & -\sin{\theta} \\
  \sin{\theta} & \sin{\theta}
\end{bmatrix}
```
Với $\theta$ là góc giữa vector cần quay và trục $Ox$, có giá trị trong đoạn $[0,2\pi]$.

Với không gian $n\ge 3$ chiều, ta định nghĩa ma trận quay Givens $G(i, j, \theta)$ như sau:
```math
G(i, j, \theta)=\begin{bmatrix}
  1 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\
  \vdots & \ddots & \vdots &  & \vdots &  & \vdots \\
  0 & \cdots & c & \cdots & -s & \cdots & 0 \\
  \vdots & & \vdots & \ddots & \vdots &  & \vdots \\
  0 & \cdots & s & \cdots & c & \cdots & 0 \\
  \vdots & & \vdots & & \vdots & \ddots & \vdots \\
  0 & \cdots & 0 & \cdots & 0 & \cdots & 1 \\
\end{bmatrix}
```
Với $c = \cos{\theta},s=\sin{\theta}$ ở các hàng và cột thứ $i$ và $j$. Với $i>j$, phần tử khác $0$ của ma trận $G(i,j,\theta$) được xác định như sau:
<ul>
  <li>$g_{kk}=1$, với $k\neq i,j$</li>
  <li>$g_{kk}=c$, với $k=i,j$</li>
  <li>$g_{ji}=-g_{ij}=-s$</li>
</ul>


Ta cũng có nhận xét: Khi thực hiện phép nhân ma trận $G(i,j,\theta)A$, chỉ hàng $i$ và $j$ của $A$ bị ảnh hưởng, do đó bài toán của trở thành: tìm $c$ và $s$ để:
```math
\begin{bmatrix}
c & -s \\
s & c
\end{bmatrix}\begin{bmatrix}
a \\
b \\
\end{bmatrix}=\begin{bmatrix}
r \\
0 \\
\end{bmatrix}
```
Trong đó $r=\sqrt{a^2+b^2}$, $a$ và $b$ là hai phần tử thuộc cùng một cột của hàng $i$ và $j$ của ma trận $A$. Rõ ràng, một bộ số $(c, s)$ thỏa mãn là $\left(\dfrac{a}{r}, \dfrac{b}{r}\right)$, tuy nhiên việc tính $r$ có thể gây ra tình trạng bị tràn số. Hiện nay nhiều ngôn ngữ lập trình sử dụng hàm `hypot`, một phương pháp tính $r$ mà không sử dụng hàm căn bậc $2$ (về hàm `hypot`, mọi người có thể tham khảo link sau: [hypot implementation](https://calhoun.nps.edu/server/api/core/bitstreams/a0926429-924e-42e7-bcd6-d077e88595c4/content?utm_source=chatgpt.com)). Trong source code, tác giả sử dụng hàm `arctan2` có trong thư viện `numpy` để tính góc $\theta$ và các giá trị $c, s$ tương ứng.
</p>
<h2>4. Thuật toán QR</h2>
