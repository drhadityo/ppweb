---
title: Word Embedding

---

## Nama : Dohan Rizqi Hadityo
## NIM : 210411100195
## Kelas : Pencarian dan Penambangan Data A

### Word Embedding
Word Embedding adalah teknik pemodelan bahasa untuk memetakan kata-kata ke dalam vektor angka-angka real. Teknik ini merepresentasikan kata atau frasa dalam ruang vektor dengan beberapa dimensi. Word embedding dapat dihasilkan menggunakan berbagai metode seperti jaringan saraf tiruan, matriks ko-okurensi, model probabilistik, dan lain-lain. Word2Vec adalah salah satu model yang digunakan untuk menghasilkan word embedding. Model ini terdiri dari jaringan saraf dengan dua lapisan yang sederhana, yaitu satu lapisan input, satu lapisan tersembunyi, dan satu lapisan output.

Word embedding bertujuan untuk mengubah kata-kata menjadi representasi numerik (vektor) yang memungkinkan mesin untuk memahami hubungan antara kata-kata dalam suatu teks. Kata-kata yang memiliki makna yang mirip akan berada lebih dekat dalam ruang vektor, yang mempermudah berbagai aplikasi NLP (Natural Language Processing) seperti pemrosesan teks, analisis sentimen, dan penerjemahan otomatis. Word2Vec adalah salah satu metode yang sangat populer untuk menghasilkan embedding ini, karena bisa menangkap hubungan semantik antar kata.

### Word2Vec
Word2Vec adalah metode yang banyak digunakan dalam pemrosesan bahasa alami (Natural Language Processing/NLP) yang memungkinkan kata-kata direpresentasikan sebagai vektor dalam ruang vektor kontinu. Word2Vec merupakan upaya untuk memetakan kata-kata ke dalam vektor berdimensi tinggi guna menangkap hubungan semantik antar kata, yang dikembangkan oleh para peneliti di Google. Prinsip utama dari Word2Vec adalah bahwa kata-kata dengan makna yang mirip seharusnya memiliki representasi vektor yang serupa. Word2Vec menggunakan dua arsitektur:

#### Python program to generate word vectors using Word2Vec
```
# importing all necessary modules
from gensim.models import Word2Vec  # Mengimpor kelas Word2Vec dari modul gensim untuk membuat model word embedding
import gensim  # Mengimpor keseluruhan pustaka gensim, yang digunakan untuk pemrosesan teks
from nltk.tokenize import sent_tokenize, word_tokenize  # Mengimpor fungsi sent_tokenize untuk memecah teks menjadi kalimat, dan word_tokenize untuk memecah kalimat menjadi kata-kata
import warnings  # Mengimpor modul warnings untuk menangani atau mengabaikan pesan peringatan dalam program
```

```
# Membaca file 'alice.txt'
sample = open("C:\\Users\\Admin\\Desktop\\alice.txt")  # Membuka file 'alice.txt' yang berada di direktori spesifik di komputer

s = sample.read()  # Membaca seluruh isi file 'alice.txt' dan menyimpannya dalam variabel 's'

# Mengganti karakter escape (newline) dengan spasi
f = s.replace("\n", " ")  # Mengganti setiap karakter newline '\n' dalam teks dengan spasi, lalu menyimpannya ke dalam variabel 'f'
```

```
data = []  # Inisialisasi daftar kosong 'data' untuk menyimpan kalimat yang telah di-tokenisasi menjadi kata-kata
 
# Melakukan iterasi melalui setiap kalimat dalam file (yang sudah diproses menjadi variabel 'f')
for i in sent_tokenize(f):  # Memecah teks 'f' menjadi kalimat-kalimat menggunakan fungsi sent_tokenize dari NLTK
    temp = []  # Inisialisasi daftar sementara 'temp' untuk menyimpan kata-kata dari satu kalimat
 
    # Tokenisasi kalimat menjadi kata-kata
    for j in word_tokenize(i):  # Memecah kalimat 'i' menjadi kata-kata menggunakan fungsi word_tokenize dari NLTK
        temp.append(j.lower())  # Menambahkan setiap kata yang telah diubah menjadi huruf kecil ke dalam daftar sementara 'temp'
 
    data.append(temp)  # Menambahkan daftar 'temp' (berisi kata-kata dari satu kalimat) ke dalam daftar 'data'
```

```
# Membuat model CBOW
model1 = gensim.models.Word2Vec(data, min_count=1,  # Membuat model Word2Vec dengan metode CBOW
                                vector_size=100,  # Ukuran vektor untuk merepresentasikan setiap kata adalah 100 dimensi
                                window=5)  # Menggunakan jendela konteks sebesar 5 kata di kiri dan kanan kata target

# Membuat model Skip Gram
model2 = gensim.models.Word2Vec(data, min_count=1,  # Membuat model Word2Vec dengan metode Skip Gram
                                vector_size=100,  # Ukuran vektor untuk merepresentasikan setiap kata adalah 100 dimensi
                                window=5,  # Menggunakan jendela konteks sebesar 5 kata di kiri dan kanan kata target
                                sg=1)  # Mengaktifkan model Skip Gram (sg=1), jika sg=0 akan menggunakan model CBOW
```

```
# Mencetak hasil
print("Cosine similarity between 'alice' " +  # Mencetak string untuk hasil perbandingan
      "and 'wonderland' - CBOW : ",  # Menunjukkan perbandingan antara kata 'alice' dan 'wonderland' dengan model CBOW
      model1.wv.similarity('alice', 'wonderland'))  # Menghitung dan mencetak nilai kemiripan kosinus antara 'alice' dan 'wonderland' menggunakan model CBOW

print("Cosine similarity between 'alice' " +  # Mencetak string untuk hasil perbandingan
      "and 'machines' - CBOW : ",  # Menunjukkan perbandingan antara kata 'alice' dan 'machines' dengan model CBOW
      model1.wv.similarity('alice', 'machines'))  # Menghitung dan mencetak nilai kemiripan kosinus antara 'alice' dan 'machines' menggunakan model CBOW

# Mencetak hasil
print("Cosine similarity between 'alice' " +  # Mencetak string untuk hasil perbandingan
      "and 'wonderland' - Skip Gram : ",  # Menunjukkan perbandingan antara kata 'alice' dan 'wonderland' dengan model Skip Gram
      model2.wv.similarity('alice', 'wonderland'))  # Menghitung dan mencetak nilai kemiripan kosinus antara 'alice' dan 'wonderland' menggunakan model Skip Gram

print("Cosine similarity between 'alice' " +  # Mencetak string untuk hasil perbandingan
      "and 'machines' - Skip Gram : ",  # Menunjukkan perbandingan antara kata 'alice' dan 'machines' dengan model Skip Gram
      model2.wv.similarity('alice', 'machines'))  # Menghitung dan mencetak nilai kemiripan kosinus antara 'alice' dan 'machines' menggunakan model Skip Gram
```

### 1. Skrip-Gram
Skip Gram: Skip gram memprediksi kata-kata konteks di sekitar dalam jendela tertentu berdasarkan kata saat ini. Lapisan input berisi kata saat ini, dan lapisan output berisi kata-kata konteks. Lapisan tersembunyi mengandung jumlah dimensi yang ingin kita gunakan untuk merepresentasikan kata yang ada di lapisan input.

Berbeda dengan CBOW yang memprediksi kata tengah dari kata-kata di sekitarnya, Skip Gram berfungsi sebaliknya, yaitu memprediksi kata-kata di sekitar berdasarkan kata target. Model ini berguna dalam menangkap hubungan semantik antar kata, terutama untuk menangani data teks yang lebih besar dan kompleks.

#### Arsitektur Skrip Gram
![arsitektur skripgram](https://hackmd.io/_uploads/BkhOpT4A0.png)


#### Contoh Skrip Gram
Sentence = Liverpool adalah klub terkuat inggris

1. Tentukan kata targetnya
    $\text{Target : "adalah"}$
    
2. Tentukan n-gram (kata sebelum dan setelah kata target)
    $n\text{-gram} = 1$

3. Ubah sentence/kalimat dan dengan one hot encodding :

    $$
    \text{Liverpool } =
    \begin{bmatrix}
    1 \\
    0 \\
    0 \\
    0 \\
    0
    \end{bmatrix}
    $$
    
    $$
    \text{adalah } =
    \begin{bmatrix}
    0 \\
    1 \\
    0 \\
    0 \\
    0
    \end{bmatrix}
    $$
    
    $$
    \text{klub } =
    \begin{bmatrix}
    0 \\
    0 \\
    1 \\
    0 \\
    0
    \end{bmatrix}
    $$
    
    $$
    \text{terkuat } =
    \begin{bmatrix}
    0 \\
    0 \\
    0 \\
    1 \\
    0
    \end{bmatrix}
    $$
    
    $$
    \text{inggris } =
    \begin{bmatrix}
    0 \\
    0 \\
    0 \\
    0 \\
    1
    \end{bmatrix}
    $$
    
4. Tentukan $\text{layer input} (X)$ atau kata target
    
    $$
    \text{adalah } =
    \begin{bmatrix}
    0 \\
    1 \\
    0 \\
    0 \\
    0
    \end{bmatrix}
    $$
    
    ukuran matrix pada $\text{layer input} (X)$ adalah $(v \times 1)$ atau $(5 \times 1)$
    
5. Tentukan representasi vector atau hidden layer
   $\text{hidden layer} = 3$
    $$
    \text{hidden layer (h) } =
    \begin{bmatrix}
    h1 \\
    h2 \\
    h3 
    \end{bmatrix}
    $$
    
6. Tentukan $W_{\text{input}}$ 
   $W_{\text{input}}$ dapat ditentukan dari $\text{hidden layer} (h)= W_{\text{input}} . \text{layer input} (x)$
   $(n \times 1) = W^{T}_{\text{input}} . (v \times 1)$
   $W^{T}_{\text{input}} = \frac{(n \times 1)}{(v \times 1)}$
   $W_{\text{input}} = (v \times n)$
   $$
   W_{\text{input}} = 
   \begin{bmatrix}
   W_{11} & W_{12} & W_{13} \\
   W_{21} & W_{22} & W_{23} \\
   W_{31} & W_{32} & W_{33} \\
   W_{41} & W_{42} & W_{43} \\
   W_{51} & W_{52} & W_{53} \\
   \end{bmatrix}
   $$

   
7. Tentukan $\text {layer output} = y$
   Ukuran matrix pada layer output sama denga layer input dikarenakan masih dalam 1 sentence yang ukuran matrixnya $(v \times 1) atau (5 \times 1)$
   
    $$
    \text{Liverpool } =
    \begin{bmatrix}
    1 \\
    0 \\
    0 \\
    0 \\
    0
    \end{bmatrix}
    $$
    $$
    \text{klub } =
    \begin{bmatrix}
    0 \\
    0 \\
    1 \\
    0 \\
    0
    \end{bmatrix}
    $$
    
8. Tentukan nilai dari $W_{\text{output}}$
   $W_{\text{output}}$ dapat ditentukan dari $\text{hidden layer}      (h)= W_{\text{input}} . \text{layer output} (x)$
   $(n \times 1) = W^{T}_{\text{output}} . (v \times 1)$
   $W^{T}_{\text{output}} = \frac{(v \times 1)}{(n \times 1)}$
   $W_{\text{output}} = (n \times v)$
   $$
   W_{\text{input}} = 
   \begin{bmatrix}
   W_{11} & W_{12} & W_{13} & W_{14} & W_{15} \\
   W_{21} & W_{22} & W_{23} & W_{24} & W_{25} \\
   W_{31} & W_{32} & W_{33} & W_{34} & W_{35} \\
   \end{bmatrix}
   $$
    
9. Hitung nilai $\text{hidden layer} (h)$ dengan membangkitkan        nilai random yang ada pada $W_{input}$
   $\text{hidden layer} (h) = \text{layer input} (x) . W_{\text{input}}$
   $h = (v \times 1).(v \times n)$
   $h = (5 \times 1).(5 \times 3)$
   $$
h =
\begin{bmatrix}
0 \\
1 \\
0 \\
0 \\
0
\end{bmatrix}
\cdot
\begin{bmatrix}
0.84 & 0.69 & 0.56 \\
0.88 & 0.31 & 0.49 \\
0.70 & 0.65 & 0.98 \\
0.62 & 0.83 & 0.63 \\
0.21 & 0.85 & 0.88 \\
\end{bmatrix}
$$

$$
h =
\begin{bmatrix}
0.88 & 0.31 & 0.49 \\
\end{bmatrix}
$$
  
10. Hitung $\hat{y}$ atau $\text{y prediksi (output prediksi)}$ dari $\text{hidden layer} (h) . W_{\text{output}}$ dengan membangkitkan nilai random dari $W_{\text{output}}$
$\hat{y} = \text{hidden layer}(h) . W_{\text{output}}$

$$
\begin{bmatrix}
0.88 & 0.31 & 0.49 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
0.34 & 0.67 & 0.21 & 0.59 & 0.58 \\
0.45 & 0.61 & 0.15 & 0.34 & 0.22 \\
0.83 & 0.40 & 0.29 & 0.61 & 0.72 \\
\end{bmatrix}
$$

$$
\hat{y} = 
\begin{bmatrix}
0.85 & 0.97 & 0.37 & 0.92 & 0.93 \\
\end{bmatrix}
$$


11. Menghitung $W^{\text{new}}_{\text{input}}$ untuk melakukan iterasi berikutnya menggunakan perhitungan dari Backpropagation
$W^{\text{new}}_{\text{input}} = W^{\text{old}}_{\text{input}} - 
\eta . x. e^{\text{T}}. W^{\text{T}}_{\text{output}}$
$W^{\text{new}}_{\text{input}} = (v \times n) - \eta. (v \times 1). (1 \times v).(v \times n)$
$W^{\text{new}}_{\text{input}} = (v \times n) -(v \times n)$

$$
W^{\text{new}}_{\text{input}} = 
\begin{bmatrix}
0.84 & 0.69 & 0.56 \\
0.88 & 0.31 & 0.49 \\
0.70 & 0.65 & 0.98 \\
0.62 & 0.83 & 0.63 \\
0.21 & 0.85 & 0.88 \\
\end{bmatrix} - 
\begin{bmatrix}
0.34 & 0.45 & 0.83 \\
0.67 & 0.61 & 0.40 \\
0.21 & 0.15 & 0.29 \\
0.59 & 0.34 & 0.61 \\
0.58 & 0.22 & 0.72 \\
\end{bmatrix}
$$

$$
W^{\text{new}}_{\text{input}} = 
\begin{bmatrix}
0.50 & 0.24 & -0.27 \\
0.21 & -0.30 & 0.09 \\
0.49 & 0.50 & 0.69 \\
0.03 & 0.49 & 0.02 \\
-0.37 & 0.63 & 0.16 \\
\end{bmatrix}
$$


12. Lakukan iterasi berikutnya dengan menggunakan $W^{\text{new}}_{\text{input}}$ yang baru
$$
h =
\begin{bmatrix}
0 \\
1 \\
0 \\
0 \\
0
\end{bmatrix}
\cdot
\begin{bmatrix}
0.50 & 0.24 & -0.27 \\
0.21 & -0.30 & 0.09 \\
0.49 & 0.50 & 0.69 \\
0.03 & 0.49 & 0.02 \\
-0.37 & 0.63 & 0.16 \\
\end{bmatrix}
$$

$$
h = \begin{bmatrix}
0.21 & -0.30 & 0.09 \\
\end{bmatrix}
$$



##### Jadi hasil representasi kata "adalah" ke dalan vektor matriks adalah 
$$
adalah = \begin{bmatrix}
0.21 & -0.30 & 0.09 \\
\end{bmatrix}
$$


### 2. CBOW
CBOW (Continuous Bag of Words): Model CBOW memprediksi kata saat ini berdasarkan kata-kata konteks di sekitarnya dalam jendela tertentu. Lapisan input berisi kata-kata konteks, dan lapisan output berisi kata yang sedang diprediksi. Lapisan tersembunyi mengandung dimensi yang ingin kita gunakan untuk merepresentasikan kata yang muncul di lapisan output.

Dalam CBOW, tujuannya adalah untuk memprediksi kata target (kata tengah) dengan melihat kata-kata di sekitarnya (kata konteks). Teknik ini sangat berguna dalam mempelajari hubungan antar kata, terutama dalam hal makna atau semantik.

#### Arsitektur CBOW
![arsitektur cbow](https://hackmd.io/_uploads/rkJcTT4R0.png)


