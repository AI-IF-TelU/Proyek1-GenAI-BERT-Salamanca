## PROPOSAL PROYEK KELOMPOK

Vibe Coding Generative AI from Scratch

## IndoMiniBERT

Generasi Judul Berita Bahasa Indonesia dengan Kontrol Gaya Menggunakan BERT

Masked Language Model yang Dibangun dari Nol

| Item | Keterangan |
| --- | --- |
| Kelompok | [Nama kelompok] |
| Anggota | 5 orang (lihat Bagian 5) |
| Kategori model | Generative Text dan NLP: BERT dari Nol (Masked Language Model) |
| Dataset | CLICK-ID, Kaggle (andikawilliam/clickid) |
| Perangkat | Google Colab atau Kaggle Notebook, GPU T4 |

## 1. Latar Belakang dan Masalah

Judul clickbait sudah jadi hal yang umum di media online Indonesia. Portal berita bergantung pada jumlah klik, sehingga banyak judul ditulis untuk memancing pembaca membuka artikel, bukan untuk menyampaikan informasi. William dan Sari (2020) menyusun dataset CLICK-ID untuk meneliti masalah ini. Dari 15.000 judul yang mereka ambil dari 12 portal berita Indonesia, sebanyak 6.290 judul atau sekitar 42% dilabeli sebagai clickbait. Angka ini menunjukkan bahwa gaya penulisan judul adalah masalah nyata yang cocok dijadikan konteks proyek model bahasa.

Proyek ini memakai masalah tersebut sebagai latar untuk BERT yang bersifat generatif. BERT adalah encoder yang awalnya dilatih untuk menebak kata yang disembunyikan dalam kalimat (Devlin et al., 2019). Wang dan Cho (2019) menunjukkan bahwa masked language model yang sama bisa menghasilkan kalimat utuh dengan cara mengisi dan mengisi ulang posisi [MASK] secara bertahap. Kami menggabungkan ide itu dengan token gaya, sehingga model bisa diminta membuat judul bergaya clickbait atau non-clickbait. Setelah itu, sebuah classifier clickbait dipakai untuk mengecek apakah judul yang dihasilkan benar-benar sesuai gaya yang diminta.

## 1.1 Tujuan

- 1. Membangun encoder BERT berukuran kecil dari nol menggunakan PyTorch, termasuk multi- head self-attention dua arah, tanpa memakai bobot pretrained apa pun.

- 2. Melakukan pretraining dengan objektif Masked Language Model (MLM) pada korpus artikel berita Indonesia.

- 3. Melatih MLM yang dikondisikan gaya sehingga mampu menghasilkan judul berita bergaya clickbait atau non-clickbait melalui decoding mask-predict.


- 4. Mengukur kualitas hasil generasi dengan classifier gaya, metrik kebaruan, dan metrik keragaman, serta mendokumentasikan seluruh proses dalam Log Interaksi LLM.

## 1.2 Mengapa BERT Termasuk Generative AI

Panduan proyek mencantumkan “BERT dari Nol (Masked Language Model)” pada kategori Generative Text dan NLP. Menebak token yang di-mask sudah merupakan tugas generatif. Proyek ini melangkah lebih jauh dengan membangkitkan judul utuh melalui decoding mask-predict, teknik yang dipakai Ghazvininejad et al. (2019) untuk generasi teks non-autoregresif. Model punya tiga mode generasi.

- Fill-mask. Pengguna menulis judul yang belum lengkap, misalnya [NCB] presiden resmikan [MASK] [MASK] di jakarta, lalu model melengkapi kata yang hilang.

- Kata kunci menjadi judul. Pengguna memberi beberapa kata kunci seperti “banjir” dan “bekasi”. Posisi lainnya di-mask dan model melengkapi judul secara bertahap.

- Generasi bebas. Model mulai dari deretan [MASK] penuh dengan satu token gaya, lalu menyusun judul dari awal.

## 2. Dataset

Proyek ini memakai CLICK-ID yang diunggah ke Kaggle oleh penyusunnya sendiri dan didukung oleh paper data di jurnal Data in Brief (William & Sari, 2020). Dataset ini terdiri dari dua bagian yang langsung sesuai dengan tahapan training kami.

| Bagian | Isi | Kegunaan di proyek |
| --- | --- | --- |
| Raw | 46.119 artikel berita dari 12 portal Indonesia Kolom: title, source, date, category, content, url | Korpus pretraining MLM Data latih tokenizer WordPiece |
| Annotated | 15.000 judul yang dilabeli 3 annotator (suara mayoritas) 6.290 clickbait dan 8.710 non-clickbait | Training MLM bergaya Training classifier clickbait |

## 3. Arsitektur Model

Model mengikuti desain encoder BERT dari Devlin et al. (2019) dengan skala yang jauh lebih kecil. Konfigurasinya mendekati BERT-Mini yang dijelaskan Turc et al. (2019), yang memang ditujukan untuk sumber daya komputasi terbatas. Semua nilai di bawah adalah titik awal dan boleh diubah saat tuning.

| Komponen | Nilai | Catatan |
| --- | --- | --- |
| Jumlah encoder layer | 4 | Attention, feed-forward, residual, LayerNorm |
| Hidden size | 256 |   |
| Attention head | 4 | 64 dimensi per head |


| Komponen | Nilai | Catatan |
| --- | --- | --- |
| Feed-forward size | 1024 | Aktivasi GELU |
| Vocabulary | Sekitar 16.000 token WordPiece | Dilatih dari korpus sendiri |
| Panjang maksimal | 128 (pretraining), 48 (judul) | Positional embedding dipelajari |
| Token khusus | [PAD] [UNK] [CLS] [SEP] [MASK] [CB] [NCB] | [CB] dan [NCB] penanda gaya |
| Output head | MLM head dan classification head | Bobot MLM head diikat ke embedding input |

Multi-head self-attention ditulis manual, tidak memakai nn.TransformerEncoder, karena membangun atensi dua arah adalah tantangan utama yang disebut di panduan. Objektif Next Sentence Prediction tidak dipakai karena Liu et al. (2019) menemukan objektif ini tidak memberi manfaat yang jelas, dan judul berita juga hanya terdiri dari satu kalimat.

## 4. Lima Step Pengerjaan

Proyek dibagi menjadi 5 step yang berurutan, dan setiap step dipegang oleh satu anggota. Walaupun berurutan, step berikutnya tetap bisa mulai lebih awal memakai data dummy atau checkpoint sementara, jadi tidak ada anggota yang menganggur menunggu step sebelumnya selesai. Setiap anggota tetap wajib mencatat Log Interaksi LLM miliknya sendiri.

*Step 1: Persiapan Data dan Tokenizer (Anggota 1)*

| Keterangan | Isi |
| --- | --- |
| Penanggung jawab | Anggota 1 (sekaligus ketua dan pengelola repo GitHub) |
| Waktu | Minggu 1 sampai 2 |
| Input | Dataset CLICK-ID dari Kaggle |
| Output | Korpus bersih, tokenizer WordPiece, Dataset class, file config.py |

- Membuat repo GitHub, struktur folder, dan config.py yang berisi semua ukuran tensor serta hyperparameter supaya modul dari setiap anggota sinkron.

- Mengunduh folder raw dan annotated, lalu membersihkan teks (hapus HTML, URL, karakter aneh, spasi ganda, ubah ke huruf kecil) dan memecah artikel menjadi kalimat.

- Membagi data annotated secara stratified menjadi 80% train, 10% validasi, dan 10% test dengan random seed tetap.

- Menghapus judul yang ada di split test dari korpus pretraining supaya tidak terjadi kebocoran data, lalu mencatat berapa kalimat yang dihapus.

- Melatih tokenizer WordPiece sekitar 16.000 token dari korpus raw dan menambahkan token [CB] dan [NCB].


- Membuat Dataset class untuk korpus pretraining dan data judul berlabel.

Kriteria selesai: satu batch dari DataLoader bisa diambil dengan shape (batch, 128) untuk pretraining dan (batch, 48) untuk judul, dan statistik dataset sudah dicatat untuk laporan.

*Step 2: Arsitektur Model BERT (Anggota 2)*

| Keterangan | Isi |
| --- | --- |
| Penanggung jawab | Anggota 2 |
| Waktu | Minggu 1 sampai 3 (mulai paralel dengan data dummy) |
| Input | config.py dan ukuran vocab dari Step 1 |
| Output | File model.py berisi encoder BERT lengkap dengan MLM head dan classification head |

- Membuat embedding layer yang menjumlahkan token embedding dan positional embedding, diikuti LayerNorm dan dropout.

- Menulis multi-head self-attention secara manual (proyeksi Q, K, V, scaled dot-product, padding mask, penggabungan head).

- Menyusun encoder block (attention, feed-forward GELU, residual connection, LayerNorm) lalu menumpuknya menjadi 4 layer.

- Membuat MLM head dengan weight tying ke embedding input (Press & Wolf, 2017) dan classification head pada token [CLS].

- Menjalankan sanity check, yaitu model harus bisa overfit pada satu batch kecil sampai loss mendekati nol.

Kriteria selesai: forward dan backward pass berjalan tanpa error, jumlah parameter tercatat, dan tes overfit satu batch berhasil.

*Step 3: Masking dan Pretraining MLM (Anggota 3)*

| Keterangan | Isi |
| --- | --- |
| Penanggung jawab | Anggota 3 |
| Waktu | Minggu 2 sampai 4 |
| Input | DataLoader dari Step 1 dan model dari Step 2 |
| Output | Checkpoint model hasil pretraining dan grafik loss |

- Membuat data collator dynamic masking. Sebanyak 15% token dipilih, lalu 80% diganti [MASK], 10% diganti token acak, dan 10% dibiarkan (Devlin et al., 2019). Token khusus tidak boleh ikut di-mask.

- Membuat unit test untuk membuktikan rasio masking sudah benar.


- Menyusun training loop pretraining dengan AdamW (Loshchilov & Hutter, 2019), linear warmup, gradient clipping, mixed precision, dan penyimpanan checkpoint berkala.

- Melakukan eksperimen minimal 3 variasi learning rate dan 2 variasi batch size, lalu memilih konfigurasi terbaik.

- Mencatat MLM loss, perplexity, serta akurasi top-1 dan top-5 token yang di-mask di setiap epoch.

Kriteria selesai: loss validasi turun secara konsisten, checkpoint terbaik tersimpan, dan model bisa menebak kata yang masuk akal pada contoh fill-mask sederhana.

*Step 4: MLM Bergaya dan Classifier Clickbait (Anggota 4)*

| Keterangan | Isi |
| --- | --- |
| Penanggung jawab | Anggota 4 |
| Waktu | Minggu 4 sampai 5 |
| Input | Checkpoint pretraining dari Step 3 dan data judul berlabel |
| Output | Model generator bergaya dan model classifier clickbait |

- Melanjutkan training MLM pada judul berlabel dengan token gaya di depan kalimat, misalnya [CLS] [CB] kamu tidak akan percaya ... [SEP]. Token gaya tidak pernah di-mask.

- Fine-tuning classification head untuk membedakan clickbait dan non-clickbait (tanpa token gaya).

- Melatih classifier dalam dua versi, yaitu dari bobot pretraining dan dari inisialisasi acak, untuk membuktikan seberapa besar kontribusi pretraining (ablation).

- Menyimpan classifier sebagai salinan terpisah yang dipakai untuk menilai hasil generasi, supaya penilaiannya tidak bias.

- Mencatat akurasi, macro F1, dan confusion matrix pada split test.

Kriteria selesai: generator bergaya dan classifier tersimpan, serta tabel perbandingan F1 versi pretrained dan versi acak sudah tersedia.

*Step 5: Generasi, Evaluasi, dan Dokumentasi (Anggota 5)*

| Keterangan | Isi |
| --- | --- |
| Penanggung jawab | Anggota 5 |
| Waktu | Minggu 4 sampai 6 |
| Input | Generator dan classifier dari Step 4, Log Interaksi LLM dari semua anggota |
| Output | Fungsi generasi, hasil evaluasi, demo, Log Interaksi LLM lengkap, laporan akhir |

- Membuat fungsi generasi mask-predict (Ghazvininejad et al., 2019). Panjang judul diambil dari distribusi panjang judul asli, lalu dalam sekitar 10 iterasi model menebak semua posisi [MASK],


- mempertahankan token paling yakin, dan me-mask ulang token yang paling tidak yakin dengan jumlah yang terus berkurang.

- Menyediakan pengaturan top-k dan temperature, lalu mengamati pengaruhnya terhadap kualitas dan keragaman judul.

- Menghitung akurasi gaya (persentase judul yang dinilai classifier sesuai gaya yang diminta), kebaruan (persentase judul yang tidak sama persis dengan data latih), serta Distinct-1 dan Distinct-2 (Li et al., 2016).

- Membuat demo ketiga mode generasi dan visualisasi attention map untuk presentasi.

- Mengumpulkan Log Interaksi LLM dari semua anggota setiap minggu dan menyusun laporan akhir.

Kriteria selesai: semua metrik dan contoh output sudah masuk laporan, demo bisa dijalankan, dan log interaksi lengkap dari tahap inisiasi, debugging, sampai optimasi.

## 5. Ringkasan Pembagian Tugas

| Anggota | Step | Minggu | Deliverable utama |
| --- | --- | --- | --- |
| Anggota 1 | Persiapan Data dan Tokenizer 1 sampai 2 |   | Korpus bersih, tokenizer, config.py, repo |
| Anggota 2 | Arsitektur Model BERT | 1 sampai 3 | model.py dan hasil tes overfit |
| Anggota 3 | Masking dan Pretraining MLM 2 sampai 4 |   | Checkpoint pretraining dan grafik loss |
| Anggota 4 | MLM Bergaya dan Classifier | 4 sampai 5 | Generator bergaya, classifier, ablation |
| Anggota 5 | Generasi, Evaluasi, Dokumentasi | 4 sampai 6 | Demo, metrik, log LLM, laporan |

## 6. Aturan Kerja Tim

- Log LLM. Setiap anggota mencatat sesi LLM-nya sendiri dengan template dari panduan (inisiasi, debugging, optimasi). Anggota 5 mengumpulkannya setiap minggu. Bobot log adalah 25% dari nilai.

- Tanpa copy-paste tutorial. Kode harus berasal dari prompt kelompok ke LLM. Dokumentasi resmi PyTorch hanya boleh dipakai sebagai referensi fungsi dasar.

- Disiplin Git. Setiap anggota bekerja di branch masing-masing dan melakukan commit dengan akun sendiri, karena riwayat commit masuk penilaian kerja sama tim.

- Satu sumber konfigurasi. Semua ukuran tensor dan hyperparameter diambil dari config.py.

- Saling paham. Setiap anggota harus bisa menjelaskan step miliknya dan memahami gambaran umum step lain untuk sesi presentasi.


## 7. Risiko dan Keterbatasan

Model dengan 4 layer yang dilatih pada puluhan ribu artikel tentu tidak akan menyamai model besar seperti IndoBERT. Beberapa judul hasil generasi kemungkinan masih terasa janggal atau kurang tepat secara tata bahasa. Hal ini akan dilaporkan apa adanya dan dibahas sebagai keterbatasan. Waktu training di GPU gratis juga terbatas, jadi checkpoint harus disimpan secara berkala. Classifier gaya juga bukan penilai yang sempurna, sehingga akurasi gaya otomatis akan dilengkapi dengan pengecekan manual pada sebagian contoh output.

## Daftar Pustaka

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171–4186.

- Ghazvininejad, M., Levy, O., Liu, Y., & Zettlemoyer, L. (2019). Mask-Predict: Parallel decoding of conditional masked language models. Proceedings of EMNLP-IJCNLP 2019, 6112–6121.

- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A diversity-promoting objective function for neural conversation models. Proceedings of NAACL-HLT 2016, 110–119.

- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.

- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations (ICLR 2019).

- Press, O., & Wolf, L. (2017). Using the output embedding to improve language models. Proceedings of EACL 2017, 157–163.

- Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). Well-read students learn better: On the importance of pre-training compact models. arXiv:1908.08962.

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems 30.

- Wang, A., & Cho, K. (2019). BERT has a mouth, and it must speak: BERT as a Markov random field language model. arXiv:1902.04094.

- William, A., & Sari, Y. (2020). CLICK-ID: A novel dataset for Indonesian clickbait headlines. Data in Brief, 32. DOI dataset: 10.17632/k42j7x2kpn.1. Kaggle: kaggle.com/datasets/andikawilliam/clickid
