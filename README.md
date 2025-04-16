# Thuật toán QR trong việc tìm trị riêng và vector riêng của ma trận

## Mục lục
- [1. Mở đầu](#1-mở-đầu)
- [2. Ý tưởng của thuật toán](#2-ý-tưởng-của-thuật-toán)
- [3. Phân rã QR](#3-phân-rã-qr)
  - [3.1. Gram-Schmidt Process](#31-gram-schmidt-process)
  - [3.2. Modified Gram-Schmidt Process](#32-modified-gram-schmidt-process)
  - [3.3. Householder Reflections](#33-householder-reflections)
  - [3.4. Givens Rotations](#34-givens-rotations)
- [4. Thuật toán QR](#4-thuật-toán-qr)
- [5. Trường hợp đặc biệt](#5-trường-hợp-đặc-biệt)
- [6. Cách chạy dự án](#6-cách-chạy-dự-án)
- [7. Ưu điểm và nhược điểm](#7-ưu-điểm-và-nhược-điểm)

## 1. Mở đầu
Trong lĩnh vực đại số tuyến tính và các ứng dụng của nó, việc tìm trị riêng (eigenvalue) và vector riêng (eigenvector) của ma trận đóng vai trò vô cùng quan trọng. Đây là những công cụ cơ bản để phân tích và xử lý nhiều vấn đề trong khoa học máy tính, vật lý, kinh tế và nhiều ngành khoa học khác. Các trị riêng và vector riêng không chỉ cung cấp thông tin về cấu trúc của ma trận mà còn giúp giải quyết các bài toán liên quan đến biến đổi ma trận, phân tích phổ, và nhiều ứng dụng thực tiễn như nén ảnh, nhận dạng mẫu, và phân tích dữ liệu.

Trong số nhiều phương pháp tìm trị riêng và vector riêng, thuật toán QR nổi bật nhờ tính hiệu quả và độ ổn định số học cao. Thuật toán này dựa trên việc phân rã QR của ma trận và được cải tiến qua nhiều phiên bản nhằm tối ưu tốc độ hội tụ cũng như khả năng xử lý ma trận lớn. Nhờ vậy, thuật toán QR trở thành một trong những công cụ mạnh mẽ nhất trong việc giải các bài toán trị riêng của ma trận.

Đề tài này tập trung nghiên cứu chi tiết thuật toán QR, từ cơ sở lý thuyết đến các bước triển khai thực tế. Bên cạnh đó, chúng ta cũng sẽ phân tích độ phức tạp, ưu điểm, nhược điểm của thuật toán, cũng như các cải tiến quan trọng giúp tăng cường hiệu quả trong thực hành. Việc nắm vững thuật toán QR không chỉ giúp hiểu sâu hơn về các phương pháp xử lý ma trận mà còn mở ra nhiều ứng dụng quan trọng trong các lĩnh vực kỹ thuật và khoa học tính toán.

## 2. Ý tưởng của thuật toán
Gọi $A$ là ma trận thực, vuông cấp $n$ mà chúng ta muốn tìm trị riêng và vector riêng. Với cách truyền thống, chúng ta sẽ cần phải giải phương trình $det(A - \lambda I) = 0$ để tìm $\lambda$, rồi từ $\lambda$ để tìm ra vector riêng. Tuy nhiên khi ma trận $A$ ngày càng lớn ($n$ càng lớn), việc tính $det(A - \lambda I)$ sẽ rất khó khăn, chưa kể đến việc phải phương trình đa thức bậc cao (vốn không có công thức nghiệm tổng quát với đa thức có bậc lớn hơn $4$).

Do đó, chúng ta có thể thử phân rã A thành tích của hai ma trận $Q$ và $R$, với $Q$ là ma trận trực giao (là ma trận vuông thỏa mãn $Q^{-1}=Q^T$), có các vector cột là các vector trực chuẩn, và $R$ là ma trận tam giác trên (chúng ta có thể ràng buộc các phần tử trên đường chéo của ma trận $R$ là số dương nếu $A$ khả nghịch để đảm bảo tính duy nhất của phân rã QR). Sau khi phân rã $A$ thành hai ma trận $Q$ và $R$, chúng ta sẽ thực hiện cập nhật $A$ và $Q'$ (là ma trận mà các vector cột chính là các vector riêng của ma trận $A$ ban đầu, với khởi tạo ban đầu $Q'=I_n$) và lặp lại các bước trên, cho đến khi $A$ hội tụ về dạng ma trận tam giác (tạm gọi là $A_{conv}$), và các phần tử nằm trên đường chéo của $A_{conv}$ chính là trị riêng của ma trận $A$ ban đầu. Với mỗi $1\le i \le n$, phần tử hàng $i$, cột $i$ của ma trận $A_{conv}$ có vector riêng là cột thứ $i$ của $Q'$.

## 3. Phân rã QR
Phân rã QR là một phần quan trọng, giúp chúng ta xây dựng chuỗi ma trận $A$ của thuật toán QR. Ý tưởng của phương pháp là ta sẽ phân rã từng cột của ma trận $A$ thành tổ hợp tuyến tính của các vector cột trực chuẩn của $Q$. Có $3$ phương pháp được sử dụng:
- Gram-Schmidt Process
- Householder Reflections
- Givens Rotations

Phân rã QR có các tính chất sau:
- Nếu $rank(A)=n$, khi đó phân rã QR là duy nhất, ngược lại sẽ có nhiều đáp án thỏa mãn cho cặp ma trận ($Q, R$)</li>
- Nếu $A$ không phải ma trận vuông (ma trận kích thước $m x n,~m\ge n$), ta vẫn định nghĩa phân rã QR, khi đó ma trận $Q$ là ma trận trực giao cấp $m$ và ma trận tam giác trên $R$ có kích thước $m x n$</li>
- Nếu $A$ là ma trận vuông cấp $n$, các phần tử của $A$ là các số phức, thì khi đó ma trận $Q$ ở đây là ma trận <em>unitary</em> (là ma trận phức có chuyển vị liên hợp - conjugate transpose bằng với nghịch đảo của nó)</li>

Trong phạm vi của đề tài, chúng ta sẽ chỉ giới hạn ma trận $A$ là ma trận thực, vuông cấp $n\ge 2$

### 3.1. Gram-Schmidt Process
Gram-Schmidt Process là quy trình dùng để xây dựng một tập hợp gồm $k$ vector ($k\ge 2$) đôi một vuông góc với nhau.

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

### 3.2. Modified Gram-Schmidt Process
Phương pháp này ra đời với mục đích: Tại bước thứ $k + 1$, loại bỏ các sai số của $u_i,~1\le i \le k$ ra khỏi việc tính $u_{k+1}$, $u_{k+2}$,.... Muốn làm được việc này, tại bước thứ $k + 1$, ta chỉ cần loại bỏ $proj_{u_i} a_k,~1\le i \le k$ ra khỏi công thức tính $u_{k+1}$ là xong.

Công thức mới sẽ là:
```math
q_i = \dfrac{a_i}{\left|\left|a_i\right|\right|},~a_j = a_j - proj_{q_j} a_i,~1 \le j \le n,~j + 1 \le i \le n
```
Khi đó $Q=\left[q_1\quad q_2\quad\cdots\quad q_n\right]$

### 3.3. Householder Reflections
Phép đối xứng Householder (hay phép biến đổi Householder) là phép biến đổi vector, trong đó ta cần biến đổi đối xứng vector $a$ cho trước thành vector $a'$ có giá trùng với giá của vector $e$ nào đó mà bảo toàn được độ lớn của vector $a$. Ở đây, ta chọn $e_1 = [1\quad 0\quad 0\quad\cdots\quad 0]$

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

### 3.4. Givens Rotations
Khác với phép biến đổi Householder, phép biến đổi Givens thực hiện biến đổi vector cột của ma trận $A$ bằng phép quay một góc $\theta$ nào đó.

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
- $g_{kk}=1$, với $k\neq i,j$
- $g_{kk}=c$, với $k=i,j$
- $g_{ji}=-g_{ij}=-s$


Ta cũng có nhận xét: Khi thực hiện phép nhân ma trận $G(i,j,\theta)A$, chỉ hàng $i$ và $j$ của $A$ bị ảnh hưởng, do đó bài toán của trở thành: Tìm $c$ và $s$ để:
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

## 4. Thuật toán QR
Trước khi bước vào phần này, chúng ta định nghĩa chuẩn $k\le 1$ của ma trận $A$, kí hiệu $\left|\left|A\right|\right|_k$ như sau:
  
```math
\left|\left|A\right|\right|_k = \sqrt[k]{\sum_{i,j} \left|a_{ij}\right|^k}
```
Khi $k\to\infty$, $\left|\left|A\right|\right|_k$ trở thành 
```math
\left|\left|A\right|\right|_{\infty}=\max\limits_{i}\sum_{j} \left|a_{ij}\right|
```
Khi đó, ta nói ma trận $A$ "hội tụ" đến ma trận $A'$ khi và chỉ khi $\left|\left|A-A'\right|\right|\to 0$ (chuẩn thường được sử dụng là chuẩn $1$ hoặc chuẩn vô cùng, trong source code, tác giả chọn chuẩn $1$, có thể điều chỉnh thông qua biến `norm_ord` trong file `global_constant.py`, muốn chọn chuẩn vô cùng thì điều chỉnh `norm_ord=np.inf`).
Định nghĩa xong điều kiện hội tụ của ma trận, ta có thể xây dựng thuật toán QR như sau:
1. Khởi tạo $Q_{eigen}=I_{n}$</li>
2. Phân rã $A_k$ (sử dụng các phương pháp phân rã ở trên)</li>
3. Cập nhật $A_k$: $A_{k+1}=R_kQ_k$ và cập nhật ma trận $Q_{eigen}=Q_{eigen}Q_k$</li>
4. Lặp cho đến khi số bước $k$ đạt giới hạn là `max_iter` hoặc khi  $\left|\left|A-A'\right|\right| < tolerance$

Khi đó, các giá trị trên đường chéo của $A_k$ hội tụ đến các trị riêng của vector $A$, còn ma trận $Q_{eigen}$ sẽ hội tụ đến ma trận $Q$ với các vector cột chính là các vector riêng của ma trận $A$.

## 5. Trường hợp đặc biệt
### 5.1. $A$ là ma trận $0_n$
Với trường hợp này, phân tích $QR$ của $A$ không phải là duy nhất, tuy nhiên ta có thể chọn một ma trận trực giao $Q$ thỏa mãn là $Q=I_n$, khi đó $R=0_n$
### 5.2 $A$ là ma trận đơn vị ($A=I_n$)
Với trường hợp này, ta có thể chọn một ma trận trực giao $Q$ thỏa mãn là $Q=I_n$, khi đó $R=I_n$ thỏa mãn ma trận tam giác trên
### 5.3 $A$ là ma trận trực giao</h3>
Vì $A$ đã là ma trận trực giao, ta có thể dễ dàng chọn được cặp ma trận ($Q, R$) thỏa mãn là ($A, I_n$)
### 5.4 $A$ là ma trận tam giác trên
Trái ngược với trường hợp $A$ là ma trận trực giao, trường hợp này ta dễ dàng chọn được cặp ma trận ($Q, R$) thỏa mãn là ($I_n, A$)
### 5.5 $A$ là ma trận đường chéo
Trong trường hợp này, $Q=I_n$ và $R=A$ (đây chính là một trường hợp đặc biệt của trường hợp $A$ là ma trận tam giác trên), tuy nhiên, cần phải chú ý dấu của các phần tử trên đường chéo của $A$ để thỏa mãn tính duy nhất của phân rã QR, bằng cách điều chỉnh dấu của các phần tử trên đường chéo của $R$ thành số dương và điều chỉnh dấu của $Q$ tương ứng sao cho $A=QR$ vẫn thỏa mãn

## 6. Cách chạy dự án
<p>Để chạy được dự án này, yêu cầu Python 3.12.5 phải được cài đặt sẵn trên máy tính (nên đề xuất cài đặt thêm VSCode hoặc PyCharm), sau đó làm theo các bước sau:<br>
  1. Nếu trên máy có cài sẵn `git`, clone project này về máy tính cá nhân bằng lệnh sau:
    
  ```
  git clone https://github.com/hieukien503/NM_Project.git
  ```
  Nếu chưa cài `git`, vào folder `NM_Project`, ấn vào nút `Code` màu xanh lá, chọn "Download ZIP". Sau khi tải xong, hãy giải nén file này ra.
  2. Vào VSCode (hoặc PyCharm), mở terminal để có thể chạy file bằng dòng lệnh (CLI - Command Line Interface)

Trước khi chạy dự án, gõ `python main.py -h` để xem các flags được thiết kế sẵn cho việc chạy dự án này, các flags đó được liệt kê như trong hình dưới đây:![Screenshot 2025-04-01 150909.png](<https://media-hosting.imagekit.io/9cb7bff5bf604ea1/Screenshot%202025-04-01%20150909.png?Expires=1838103041&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=MYoch08jiUybBagSsIUoJDYpsFM95XTQ4~GYPgS9NfgRKwIxbSC8m4e9bKOMss2x4zuM5~m6PIkXIiJ8F0rH0YHjfO4nJVs8jZ-EO97pKWByrxxJlqmW8rLavoPfLoOoBK4eQvpU1cUI8yaysGT1GvygTIGDk0Z4EMLWvZyoBLmG1Q4lOq~lYqEFzDEXaPkFMH43yGaZPl4QTvjPlZDd0rG4bb35utU~csEVxS~ca3kXYKDWSw~Yi8lyPMF81sffY1mkWaf674GboodSdr7x8ZEpU-uPdpDH0Pms4IpntzvOzhmopgW-kDFeidYtbJJSzVZmVH4hqJqTHzOTy2YBXg__>)

Các flags sau là bắt buộc:
<ul>
  <li>
    `--run`: Đây là flag dùng để chạy dự án, không nhận tham số đầu vào, nhưng phải có flag `--eigens` (để tìm trị riêng và vector riêng) hoặc `--qr_decompo` (để chạy phân rã QR)</li>
  <li>`--input`: Nhận $1$ tham số đầu vào là đường dẫn đến file input chứa ma trận $A$ (ở đây file chứa ma trận $A$ có tên là `test.txt`)</li>
</ul>

<em>Note: khi chạy flag `--run` đi kèm với flag `--eigens`, sẽ có thông báo chọn thêm các mode để chạy và so sánh kết quả, với 1 là so sánh với phương pháp dùng đa thức đặc trưng truyền thống, 2 và 3 là so sánh kết quả với các hàm có sẵn trong các thư viện của Python. Còn nếu chạy flag `--run` kèm với `--qr_decompo`, chương trình sẽ đánh giá hiệu năng của các phương pháp phân rã QR so với thư viện `numpy` với hàm `numpy.linalg.qr`</em>
</p>

## 7. Ưu điểm và nhược điểm
### 7.1. Ưu điểm
#### 7.1.1. Tính linh hoạt
Thuật toán QR và phân rã QR có thể áp dụng cho cả ma trận thực và phức, ma trận vuông hoặc không vuông.
#### 7.1.2. Tính chính xác
Nếu $A$ là ma trận đối xứng và tất cả phần tử là số thực, các trị riêng được sinh ra từ thuật toán QR có độ chính xác rất cao do sai số của chúng sau mỗi lần lặp càng ngày càng nhỏ.
#### 7.1.3. Hiệu quả với các Dense matrix
Khác với <em>sparse matrix</em>, dense matrix là ma trận mà phần lớn các phần tử khác $0$, và thuật toán QR rất phù hợp cho việc tìm trị riêng của các ma trận loại này, đặc biệt càng hiệu quả nếu ma trận đó là đối xứng (trong không gian thực), hoặc ma trận Hermitian (trong không gian phức, là ma trận bằng với chuyển vị liên hợp của ma trận đó).
### 7.2. Nhược điểm
Trong dự án này, với giả thiết $A, Q, R$ là các ma trận thực, thuật toán QR có những nhược điểm sau:
#### 7.2.1. Gặp khó khăn trong trường hợp trị riêng là số phức
Nếu phương trình $det(A-\lambda I)=0$ cho ra nghiệm phức, thuật toán QR tỏ ra không hiệu quả (do thuật toán QR trả về trị riêng là các phần tử nằm trên đường chéo của $A_k$, vốn là các số thực). Ta có thể mở rộng thuật toán QR và các hàm phân rã QR cho không gian phức để giải quyết vấn đề này (xem trong file `complex_qr_algorithm.py`)
#### 7.2.2. Chi phí tính toán lớn
Thuật toán phân rã QR có độ phức tạp (về thời gian) là $O(n^3)$ ($n$ là kích thước của ma trận), dẫn đến việc thuật toán QR có độ phức tạp về thời gian là $O(mn^3)$), với $m$ là số lần lặp (nếu $m \ll n$ do hội tụ quá nhanh thì độ phức tạp sẽ là $O(n^3)$).
#### 7.2.3. Tốn bộ nhớ
Nếu $A$ không có dạng [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) và kích thước ma trận lớn ($>2000$), bộ nhớ cần thiết cho thuật toán sẽ rất lớn
#### 7.2.4. Vector riêng rất dễ bị nhiễu loạn
Mặc dù các giá trị riêng được tính toán chính xác, các vectơ riêng có thể nhạy cảm với nhiễu loạn (noises), đặc biệt đối với các ma trận có các giá trị riêng cách đều nhau.
#### 7.2.5 Có thể không hội tụ
Nếu ma trận $A$ không chuẩn (Non-normal matrix, là ma trận thỏa $AA^T\neq A^TA$), thuật toán QR có thể không hội tụ hoặc sinh ra các vector riêng không chính xác.
