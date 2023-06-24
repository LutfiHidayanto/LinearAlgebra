# Kalkulator Matriks untuk Aljabar Linear

Sebuah program Python yang melakukan berbagai perhitungan matriks, termasuk pemecahan sistem persamaan linear, dekomposisi nilai singular, eigennilai dan eigenvector, invers matriks, memeriksa apakah matriks diagonal, karakteristik polinomial, dan diagonalisasi matriks.

## Persyaratan

- Python 3.x
- pustaka numpy
- pustaka sympy

## Instalasi

1. Klon repository atau unduh berkas program.
2. Pasang pustaka yang diperlukan dengan menjalankan perintah berikut:
   ```
   pip install numpy sympy
   ```
   atau
   ```
   pip3 install numpy sympy
   ```

## Penggunaan

1. Buka terminal atau command prompt.
2. Arahkan ke direktori di mana berkas program berada.
3. Jalankan program dengan mengeksekusi perintah berikut:
   ```
   python matrix.py
   ```
4. Ikuti petunjuk yang ditampilkan di layar untuk memilih operasi yang diinginkan dan memberikan input yang diperlukan.
5. Lihat hasilnya di terminal atau command prompt.
6. Program juga akan menghasilkan sebuah berkas bernama "matrix_output.txt" yang berisi perhitungan yang dilakukan.

## Menu Program

Program ini menyediakan opsi berikut:

1. Sistem Persamaan Linear: Menyelesaikan sistem persamaan linear.
2. Dekomposisi Nilai Singular (SVD): Melakukan dekomposisi nilai singular pada sebuah matriks.
3. Eigennilai dan Eigenvector: Menghitung eigennilai dan eigenvector dari sebuah matriks.
4. Invers Matriks: Menghitung invers dari sebuah matriks.
5. Periksa Apakah Matriks Diagonal: Memeriksa apakah sebuah matriks diagonal.
6. Karakteristik Polinomial: Menghitung karakteristik polinomial dari sebuah matriks.
7. Diagonalisasi Matriks: Diagonalisasi sebuah matriks.
8. Sistem Persamaan Linear Kompleks: Menyelesaikan sistem persamaan linear kompleks.
9. Keluar: Keluar dari program.

## Tutorial Menguji Test case

1. Buka file "Test cases.txt"
2. Perhatikan tulisan berikut
   ```
   *Gauss Jordan
   Input:
   1
   y
   4
   4
   1 1 -1 -1
   2 5 -7 -5
   2 -1 1 3
   5 2 -4 2
   1 -2 4 6
   1
   ```
3. Copy paste
   ```
   1
   y
   4
   4
   1 1 -1 -1
   2 5 -7 -5
   2 -1 1 3
   5 2 -4 2
   1 -2 4 6
   1
   ```
   ke dalam terminal. Paste menggunakan ctrl + shift + v jika menggunakan windows os.
4. Jika output belum muncul, klik Enter
5. Berikut adalah contoh output:
   ```
   Output:
   The system of equations is underdetermined with infinite solutions.
   Using Gauss-Jordan Method
   Solution: {'x1': 1.83333333333333 - 0.666666666666667*x4, 'x2': 2.66666666666667*x4 - 0.0833333333333333, 'x3': x4 + 0.75, 'x4': x4}
   ```
