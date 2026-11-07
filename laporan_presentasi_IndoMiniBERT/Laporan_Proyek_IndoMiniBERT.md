# Laporan Proyek IndoMiniBERT

## Generasi Judul Berita Bahasa Indonesia dengan Kontrol Gaya

| Keterangan | Detail |
|---|---|
| Mata kuliah/kegiatan | Proyek 1 — Generative AI / NLP |
| Kelompok | Salamanca |
| Dataset | CLICK-ID |
| Tanggal penyusunan | 7 November 2026 |

## Identitas Anggota

Nama dan NIM berikut dicantumkan sesuai slide presentasi.

| No. | Nama | NIM |
|---:|---|---|
| 1 | Muhammad Hanif Pearlyaradja | 103012340237 |
| 2 | Muhammad Zayyad Azizan | 103012340211 |
| 3 | Calvin Nathaniel | 103012340285 |
| 4 | Nesha Rizqika Dwinity | 103012340359 |
| 5 | Almer Akbar Alqifari | 103012340484 |

## Abstrak

IndoMiniBERT adalah proyek pemodelan bahasa untuk mengolah judul berita Bahasa Indonesia dan mengeksplorasi generasi teks dengan kendali gaya *clickbait* dan *non-clickbait*. Proyek mengimplementasikan encoder BERT kecil menggunakan PyTorch, melatih objektif *Masked Language Modeling* (MLM), menambahkan token gaya `[CB]` dan `[NCB]`, serta membandingkan classifier yang menggunakan encoder hasil pretraining dengan classifier berinisialisasi acak. Rancangan presentasi melanjutkan alur tersebut dengan algoritma generasi *Mask-Predict*.

Artefak proyek mencatat 14.958 judul berlabel setelah deduplikasi, tokenizer WordPiece 16.000 token, serta evaluasi classifier pada 1.496 judul test. Pada berkas evaluasi Step 4, varian pretrained memperoleh akurasi 79,61% dan macro F1 78,58%, sedikit lebih tinggi daripada varian random-init (78,68% dan 77,79%). Namun, training pretraining yang tersimpan hanya menyelesaikan 1 dari 10 epoch yang diminta. Selain itu, angka generasi pada slide belum dapat diverifikasi dari kode yang tersedia di folder proyek. Karena itu, laporan ini membedakan hasil yang tercatat pada artefak proyek dari hasil yang hanya dicantumkan pada presentasi.

## 1. Pendahuluan

Judul berita digital menggunakan rentang gaya penulisan yang luas. Sebagian judul menyampaikan informasi secara langsung, sementara sebagian lainnya menggunakan rasa ingin tahu atau unsur emosional untuk menarik klik. Proyek ini menggunakan klasifikasi *clickbait* sebagai konteks pengembangan model Bahasa Indonesia berskala kecil.

BERT pada umumnya merupakan encoder dua arah yang dilatih untuk memprediksi token yang disembunyikan. Dengan memasukkan token `[MASK]` dan memprediksi ulang posisi tersebut, model MLM juga dapat digunakan untuk melengkapi teks. IndoMiniBERT mengembangkan pendekatan ini dengan token gaya sebagai kondisi generasi: `[CB]` untuk gaya *clickbait* dan `[NCB]` untuk gaya *non-clickbait*.

### Tujuan

1. Membangun encoder BERT mini dari awal menggunakan PyTorch dan multi-head self-attention dua arah.
2. Melatih model dengan objektif MLM pada korpus berita Bahasa Indonesia.
3. Mengondisikan MLM dengan token gaya dan membandingkan classifier berbobot pretrained dengan classifier random-init.
4. Mengeksplorasi generasi judul melalui *Mask-Predict* dan mengevaluasi karakteristik hasilnya.

## 2. Data dan Prapemrosesan

Proyek menggunakan CLICK-ID, yang terdiri atas korpus artikel berita mentah dan kumpulan judul berlabel. Berkas statistik proyek mencatat 15.000 baris judul berlabel pada awal pemrosesan. Deduplikasi menghapus 34 baris, termasuk judul dengan label yang bertentangan, sehingga tersisa 14.958 judul: 6.276 *clickbait* dan 8.682 *non-clickbait*.

Judul dibagi secara stratified dengan seed 42:

| Split | Jumlah | Clickbait | Non-clickbait |
|---|---:|---:|---:|
| Train | 11.966 | 5.021 | 6.945 |
| Validasi | 1.496 | 628 | 868 |
| Test | 1.496 | 627 | 869 |

Untuk mengurangi kebocoran data, judul pada split test dikeluarkan dari korpus pretraining. Log Step 1 mencatat 1.510 kalimat judul dan 8 kalimat isi artikel dibuang. Korpus pretraining yang dicatat berisi 45.497 artikel train dan 928 artikel validasi. Tokenizer WordPiece dilatih dengan ukuran kosakata 16.000 token. Panjang judul rata-rata tercatat 12,47 token dan maksimum 28 token; panjang masukan judul dibatasi 48 token.

## 3. Arsitektur dan Metode

### 3.1 Encoder IndoMiniBERT

Implementasi `src/model.py` membangun encoder BERT mini dengan konfigurasi berikut:

| Komponen | Konfigurasi |
|---|---:|
| Jumlah encoder layer | 4 |
| Hidden size | 256 |
| Attention head | 4 (64 dimensi per head) |
| Ukuran feed-forward | 1.024 |
| Kosakata | 16.000 token |
| Panjang maksimum pretraining | 128 token |
| Panjang maksimum judul | 48 token |

Self-attention multi-head ditulis manual dengan proyeksi query, key, dan value. Feed-forward menggunakan aktivasi GELU. Head MLM mengikat bobot proyeksi keluar dengan embedding token (*weight tying*). Token khusus mencakup `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `[CB]`, dan `[NCB]`.

### 3.2 Pretraining MLM

Pada MLM, sebagian token dipilih secara dinamis untuk diprediksi. Rancangan proyek mengikuti pola masking BERT: 15% token dipilih, lalu sebagian besar diganti `[MASK]`, sebagian kecil diganti token acak, dan sisanya dibiarkan. Token khusus dilindungi agar tidak menjadi target prediksi.

### 3.3 MLM Berkondisi Gaya dan Classifier

Untuk MLM bergaya, urutan masukan judul menggunakan bentuk `[CLS] [CB]/[NCB] judul [SEP]`. Token gaya pada posisi setelah `[CLS]` tidak boleh dimasking. Classifier memprediksi label *clickbait* atau *non-clickbait* menggunakan representasi `[CLS]`; classifier tidak menerima token gaya sebagai masukan.

Eksperimen ablation membandingkan dua varian:

- **Random-init:** seluruh encoder dan head classifier dimulai dari bobot acak.
- **Pretrained:** encoder memuat bobot dari checkpoint pretraining, sedangkan head classifier diinisialisasi untuk tugas klasifikasi.

### 3.4 Generasi *Mask-Predict* pada Presentasi

Presentasi menggambarkan alur generasi dalam empat tahap: inisialisasi template dengan token gaya, prediksi awal secara berurutan, pengisian ulang posisi dengan probabilitas rendah, lalu penerapan batasan sintaksis. Tiga mode yang ditampilkan adalah generasi bebas, template prompt, dan generasi dengan kata kunci.

Deskripsi tersebut merupakan rancangan yang disampaikan pada presentasi. Pada folder proyek yang ditelaah, belum ditemukan modul `src/generate.py` atau `src/evaluate.py`; oleh sebab itu implementasi dan hasil generasi tersebut tidak dapat diverifikasi dari kode proyek yang tersedia.

## 4. Hasil dan Pembahasan

### 4.1 Proses Training yang Tercatat

Berkas `reports/pretrain_mlm_metrics.json` mencatat training pada CPU dengan 10 epoch diminta, tetapi baru 1 epoch selesai. Pada epoch tersebut, loss train tercatat 7,204 dan loss validasi 6,335; akurasi token MLM validasi tercatat 16,19%. Dengan demikian, angka tersebut menunjukkan progres training yang terbatas, bukan hasil training penuh sesuai konfigurasi.

Berkas `reports/style_mlm_metrics.json` juga mencatat satu epoch training MLM bergaya: loss train 35,855 dan loss validasi 26,474. Angka loss ini dilaporkan sebagaimana tersimpan; nilainya tidak dibandingkan langsung dengan loss pretraining karena keduanya berasal dari tahap dan konfigurasi tugas berbeda.

### 4.2 Evaluasi Classifier Clickbait

Berdasarkan `reports/step4_evaluation.json`, classifier dievaluasi pada 1.496 judul test. Hasil utamanya:

| Varian | Akurasi | Macro F1 | F1 berbobot |
|---|---:|---:|---:|
| Random-init | 78,68% | 77,79% | 78,51% |
| Pretrained | 79,61% | 78,58% | 79,34% |

Macro F1 pretrained lebih tinggi sekitar 0,79 poin persentase dibandingkan random-init. Hasil ini menunjukkan keunggulan kecil pada evaluasi tersimpan, tetapi belum cukup untuk menyimpulkan bahwa keunggulan tersebut signifikan secara statistik.

Matriks konfusi pada laporan evaluasi (urutan label: *non-clickbait*, lalu *clickbait*) adalah:

| Varian | TN (non-clickbait benar) | FP | FN | TP (clickbait benar) |
|---|---:|---:|---:|---:|
| Random-init | 738 | 131 | 188 | 439 |
| Pretrained | 760 | 109 | 196 | 431 |

Varian pretrained mengklasifikasikan lebih banyak contoh *non-clickbait* dengan benar, tetapi pada matriks ini jumlah *clickbait* yang terdeteksi benar lebih rendah daripada random-init. Hasil per kelas patut diperhatikan karena akurasi keseluruhan saja tidak menggambarkan perbedaan tersebut.

Evaluasi *fill-mask* yang tersimpan menampilkan enam contoh dan tidak ada prediksi yang sepenuhnya cocok dengan token asli pada posisi yang dimasking. Contoh ini menunjukkan bahwa kemampuan prediksi token pada tahap tersebut masih terbatas.

### 4.3 Angka Generasi pada Slide

Slide evaluasi kualitas teks mencantumkan kebaruan (*novelty*) 100%, *Distinct-2* 0,885, dan median perplexity 192,5. Slide juga memuat contoh judul untuk kedua gaya. Angka dan contoh tersebut disajikan di sini sebagai klaim pada presentasi, bukan sebagai hasil yang dapat direproduksi dari artefak proyek yang tersedia, karena skrip generasi, skrip evaluasi terkait, dan rincian prosedur penghitungan metrik tidak ditemukan pada folder proyek yang ditelaah.

## 5. Kesesuaian Presentasi dengan Artefak Proyek

Bagian presentasi tentang CLICK-ID, pencegahan kebocoran data, tokenizer WordPiece, konfigurasi BERT mini, MLM, conditioning gaya, dan ablation classifier selaras dengan konfigurasi, kode, serta laporan proyek. Sebaliknya, alur lengkap generasi *Mask-Predict*, klaim penjagaan sintaksis, dan metrik kualitas generasi belum dapat ditautkan ke berkas implementasi dan keluaran yang tersedia.

Terdapat pula beberapa versi angka classifier di berkas proyek dan log historis. Laporan ini menggunakan `reports/step4_evaluation.json` sebagai sumber utama tabel evaluasi karena berkas tersebut secara eksplisit mencatat metrik test, jumlah sampel, matriks konfusi, dan perbandingan kedua varian. Angka pada log historis atau `reports/style_classifier_ablation.json` tidak dicampurkan dengan angka tersebut karena nilainya berbeda. Sebelum angka diserahkan sebagai hasil final, checkpoint dan proses yang menghasilkan setiap berkas sebaiknya disamakan dan didokumentasikan.

## 6. Keterbatasan dan Rekomendasi

1. **Training belum tuntas.** Pretraining yang tercatat baru menyelesaikan 1 dari 10 epoch; capaian model dapat berubah jika training dilanjutkan.
2. **Hasil perlu direproduksi.** Laporan metrik merujuk ke checkpoint dengan path lokal, tetapi checkpoint tersebut tidak tersedia dalam folder proyek yang ditelaah.
3. **Generasi belum terverifikasi.** Implementasi *Mask-Predict* dan kode evaluasi novelty, *Distinct-2*, serta perplexity tidak ditemukan di folder `src`.
4. **Ada perbedaan antarartefak.** Angka classifier di laporan evaluasi, file ablation, dan log historis tidak sepenuhnya sama. Perlu dipilih satu run/checkpoint final, lalu metrik diperbarui secara konsisten.
5. **Evaluasi generasi perlu diperjelas.** Untuk menguatkan klaim presentasi, sertakan kode, contoh masukan dan keluaran, definisi serta baseline tiap metrik, dan pemeriksaan kualitas manual.
6. **Keterbatasan model kecil.** Encoder empat layer dan korpus terbatas berpotensi menghasilkan prediksi token atau judul yang kurang lancar; keluaran generatif perlu ditinjau sebelum digunakan.

## 7. Kesimpulan

IndoMiniBERT menunjukkan alur kerja yang relevan untuk proyek NLP generatif: pemrosesan dataset CLICK-ID, tokenizer WordPiece, encoder BERT mini dari awal, pretraining MLM, conditioning gaya, dan evaluasi classifier. Pada hasil test yang tercatat di `step4_evaluation.json`, varian pretrained sedikit mengungguli random-init pada akurasi dan macro F1. Namun, pretraining belum menyelesaikan seluruh epoch, prediksi *fill-mask* pada contoh tersimpan masih lemah, dan hasil generasi yang ditampilkan di presentasi belum dapat direproduksi dari kode proyek yang tersedia.

Langkah lanjutan yang disarankan adalah menyelesaikan training, mengarsipkan checkpoint dan konfigurasi run, menyelaraskan seluruh laporan metrik, lalu melengkapi serta menguji implementasi *Mask-Predict* dan evaluasi kualitas teks. Dengan pelaporan tersebut, kesimpulan tentang generasi judul bergaya dapat didukung bukti yang dapat diperiksa dan diulang.

## Daftar Pustaka

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Ghazvininejad, M., Levy, O., Liu, Y., & Zettlemoyer, L. (2019). *Mask-Predict: Parallel Decoding of Conditional Masked Language Models*. EMNLP-IJCNLP.
- Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *Well-Read Students Learn Better: On the Importance of Pre-Training Compact Models*. arXiv:1908.08962.
- William, A., & Sari, Y. (2020). *CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines*. Data in Brief, 32. https://doi.org/10.17632/k42j7x2kpn.1

## Berkas Pendamping

- Slide presentasi: [IndoMiniBERT_Presentasi.pdf](./IndoMiniBERT_Presentasi.pdf)
- Notebook Step 4: [step4_train_and_evaluate.ipynb](../notebooks/step4_train_and_evaluate.ipynb)
