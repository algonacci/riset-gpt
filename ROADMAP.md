# Roadmap Grinding Language Model

Target utama proyek ini adalah memahami language model dari first principles,
lalu berkembang dari Nano/Tiny Language Model buatan sendiri menuju Small
Language Model yang berguna.

## Arah Utama

Proyek berjalan dalam dua jalur paralel:

1. **Jalur riset:** melatih model kecil dari nol untuk memahami setiap komponen.
2. **Jalur produk:** mengadaptasi model pretrained 0.5B-3B menjadi asisten Indonesia.

Istilah Nano Language Model tidak memiliki batas resmi. Dalam roadmap ini:

```text
Nano/Tiny LM : < 1B parameter
Small LM     : 1B-15B parameter
```

Parameter bukan satu-satunya target. Metrik yang lebih penting adalah kemampuan
per parameter, token training, FLOP, watt, latency, dan biaya inference.

## Fase 1: Fondasi

- [x] Membuat character-level tokenizer.
- [x] Membuat bigram language model.
- [x] Memahami input-target next-token prediction.
- [x] Membuat batch dan context window.
- [x] Mengimplementasikan causal self-attention.
- [x] Mengimplementasikan multi-head attention.
- [x] Mengimplementasikan feed-forward, residual connection, dan LayerNorm.
- [x] Melatih GPT mini menggunakan CUDA.
- [x] Mengganti character tokenization dengan subword `tiktoken`.
- [ ] Memahami setiap bentuk tensor dalam forward pass.
- [ ] Menulis pengujian sederhana untuk tokenizer, batch, masking, dan model.

## Fase 2: Tokenizer Indonesia

- [ ] Mengumpulkan korpus Indonesia dengan lisensi yang jelas.
- [ ] Membersihkan, menormalisasi, dan mendeduplikasi teks.
- [ ] Mempertahankan istilah Inggris dan code-switching yang alami.
- [ ] Melatih byte-level BPE atau unigram tokenizer sendiri.
- [ ] Membandingkan vocabulary 4k, 8k, 16k, dan 32k.
- [ ] Mengukur token per karakter/byte untuk Indonesia dan Inggris.
- [ ] Menambahkan special token untuk chat, dokumen, dan tool call.

## Fase 3: GPT Modern

- [ ] Mengganti learned position embedding dengan RoPE.
- [ ] Mengganti LayerNorm dengan RMSNorm.
- [ ] Mengganti ReLU feed-forward dengan SwiGLU.
- [ ] Menerapkan weight tying antara token embedding dan LM head.
- [ ] Mempelajari Grouped Query Attention.
- [ ] Menggunakan scaled dot-product/Flash Attention bila tersedia.
- [ ] Menambahkan KV cache untuk generation.
- [ ] Memastikan implementasi modern mengalahkan baseline pada compute yang sama.

## Fase 4: Training Pipeline

- [ ] Menyimpan model, optimizer, step, dan konfigurasi dalam checkpoint.
- [ ] Mendukung resume training.
- [ ] Menggunakan mixed precision (`bf16` atau `fp16`).
- [ ] Menambahkan gradient clipping.
- [ ] Menambahkan warmup dan learning-rate scheduler.
- [ ] Menambahkan gradient accumulation.
- [ ] Merekam train loss, validation loss, perplexity, waktu, dan VRAM.
- [ ] Memisahkan konfigurasi, training, evaluation, dan inference.
- [ ] Menjaga eksperimen reproducible dengan seed dan config snapshot.

## Fase 5: Scaling Nano LM

Milestone model dari nol:

```text
NanoID-1M
NanoID-10M
NanoID-30M
NanoID-100M
```

- [ ] Menentukan dataset/token budget untuk setiap ukuran.
- [ ] Membandingkan loss terhadap parameter, token, dan compute.
- [ ] Menghindari model terlalu besar untuk jumlah data yang tersedia.
- [ ] Membuat benchmark bahasa Indonesia privat.
- [ ] Menguji generalisasi, bukan hanya memorisasi training set.
- [ ] Melakukan ablation untuk tokenizer, RoPE, RMSNorm, dan SwiGLU.

Target lanjutan adalah model Indonesia 100M-500M yang efisien. Model skala ini
tidak harus menjadi asisten frontier; nilainya adalah riset, efisiensi, dan
pemahaman training dari nol.

## Fase 6: Instruction dan Reasoning

- [ ] Menyusun format percakapan system-user-assistant.
- [ ] Melakukan supervised fine-tuning setelah base model belajar bahasa.
- [ ] Menambahkan data instruction-following Indonesia.
- [ ] Membangun evaluasi factuality dan calibrated uncertainty.
- [ ] Menggunakan tugas yang dapat diverifikasi untuk matematika dan coding.
- [ ] Mempelajari rejection sampling dan preference optimization.
- [ ] Mempelajari DPO sebelum reinforcement learning yang lebih kompleks.
- [ ] Mengeksplorasi GRPO atau outcome-reward training.

### Preference optimization

Preference optimization mengajarkan jawaban mana yang lebih baik, bukan hanya
meniru satu jawaban referensi. Metode yang perlu dipelajari:

- DPO.
- RLHF.
- RLAIF.
- GRPO dan variasinya.

Target post-training mencakup:

- Helpfulness.
- Correctness.
- Instruction following.
- Calibrated uncertainty: yakin ketika benar dan mengaku tidak tahu ketika bukti kurang.
- Safety.
- Style dan konsistensi karakter model.
- Tidak mengarang ketika tidak mengetahui jawabannya.

Post-training bukan kosmetik. Tahap ini sangat menentukan karakter, kegunaan,
dan keandalan base model saat menjadi asisten.

## Fase 7: Tools dan Agent

Model frontier bukan hanya model weights, tetapi sebuah sistem:

```text
LLM
+ web search
+ code execution
+ database
+ filesystem
+ memory
+ tool router
+ context management
+ permission system
```

Model perlu dilatih untuk:

- Memilih tool yang tepat atau memutuskan bahwa tool tidak diperlukan.
- Membuat nama tool dan argumen terstruktur yang valid.
- Membaca serta mengintegrasikan hasil tool ke jawaban.
- Memperbaiki kegagalan tool atau argumen yang salah.
- Berhenti ketika tujuan telah tercapai.
- Tidak memalsukan eksekusi maupun hasil tool.

Kemampuan ini memerlukan dataset trajectory yang menyimpan keputusan, tool
call, hasil eksekusi, recovery, dan jawaban akhir; bukan hanya pasangan
pertanyaan-jawaban.

- [ ] Mengajarkan output tool call terstruktur.
- [ ] Menambahkan calculator, code execution, search, dan retrieval.
- [ ] Melatih model membaca hasil tool dan memperbaiki kegagalan.
- [ ] Mencegah model memalsukan hasil tool.
- [ ] Menambahkan context management dan memory dengan batas privasi.
- [ ] Mengevaluasi keberhasilan tugas end-to-end, bukan hanya kualitas teks.

## Fase 8: Long Context dan Memory

Context window besar tidak otomatis berarti model mampu menggunakan seluruh
isinya. Kemampuan yang harus dikejar:

- Retrieval akurat dari awal, tengah, dan akhir konteks.
- Tidak kehilangan system prompt dan instruksi penting.
- Kompresi dan peringkasan konteks tanpa membuang fakta kritis.
- KV cache serta state management yang efisien.
- Retrieval-augmented generation untuk pengetahuan eksternal.
- Memory lintas sesi dengan provenance, izin, masa berlaku, dan privasi.

Ukuran context dan kemampuan memanfaatkan context adalah dua hal berbeda dan
harus dievaluasi secara terpisah.

## Fase 9: Inference System

Kualitas produk juga ditentukan oleh strategi inference:

- Temperature.
- Top-p dan top-k sampling.
- Repetition penalty.
- KV cache.
- Speculative decoding.
- Quantization.
- Continuous batching.
- Routing antar-model.
- Test-time compute.

Reasoning model modern dapat menukar compute saat inference dengan kualitas:

```text
lebih banyak token berpikir
+ beberapa kandidat jawaban
+ verifier
+ pemilihan jawaban terbaik
```

Inference perlu diukur sebagai trade-off kualitas, latency, throughput, memori,
energi, dan biaya; bukan hanya kecepatan menghasilkan token.

## Fase 10: Evaluation dan Feedback Loop

Evaluation adalah sistem kendali pengembangan model. Tanpa evaluator yang baik,
perubahan loss atau benchmark belum tentu berarti model lebih berguna.

- [ ] Membuat baseline sebelum setiap perubahan model atau dataset.
- [ ] Memisahkan validation, public benchmark, dan private benchmark.
- [ ] Menguji language modeling, pengetahuan, matematika, coding, multilingual,
  instruction following, factuality, long context, tool use, dan safety.
- [ ] Menyimpan prompt, output, skor, latency, konfigurasi, dan versi checkpoint.
- [ ] Membuat failure taxonomy agar kesalahan dapat dikelompokkan dan diperbaiki.
- [ ] Menggunakan compiler, unit test, calculator, atau verifier objektif bila memungkinkan.
- [ ] Menguji tugas produk nyata secara end-to-end.
- [ ] Menjaga benchmark privat agar model tidak hanya menghafal evaluasi publik.

Loop utama yang dikejar:

```text
train -> evaluate -> kelompokkan kegagalan -> perbaiki data/resep
      -> train lagi -> verifikasi regresi
```

## Jalur Produk SLM

Jalur ini menggunakan model pretrained agar dapat menghasilkan produk sebelum
pretraining model miliaran parameter dari nol menjadi realistis.

- [ ] Memilih base model open-source 0.5B-3B.
- [ ] Menguji baseline bahasa Indonesia sebelum fine-tuning.
- [ ] Melakukan continued pretraining dengan korpus Indonesia.
- [ ] Melakukan instruction tuning dengan LoRA/QLoRA.
- [ ] Menambahkan tool use dan retrieval.
- [ ] Melakukan quantization 4-bit untuk deployment lokal/edge.
- [ ] Membandingkan hasil dengan NanoID buatan sendiri.

## Evaluasi Wajib

- Language-modeling loss dan perplexity.
- Kemampuan bahasa Indonesia dan code-switching.
- Pengetahuan, matematika, coding, dan instruction following.
- Factuality dan kecenderungan hallucination.
- Long-context retrieval.
- Tool-use success rate.
- Latency, throughput, VRAM, energi, dan biaya.
- Benchmark privat dan tugas nyata agar tidak hanya mengejar skor publik.

## Prinsip

```text
Prediksi token yang kuat
+ representasi yang baik
+ data curriculum
+ reasoning yang dapat diverifikasi
+ post-training
+ tool use
+ evaluation loop
+ inference system
```

Prioritas terdekat: buat tokenizer BPE Indonesia, siapkan dataset yang layak,
dan bangun checkpoint serta evaluasi sebelum memperbesar model.

## Catatan: Jika Compute Bukan Masalah

Secara prinsip, frontier model dapat dibangun jika compute tersedia, tetapi
compute dan data saja tidak menjamin hasil frontier. Kemampuan akhir merupakan
hasil perkalian banyak komponen:

```text
Capability =
architecture
x data
x optimization
x post-training
x evaluation
x inference system
x engineering reliability
```

Satu komponen yang lemah dapat menahan atau menggagalkan keseluruhan sistem.

### Data

Yang diperlukan bukan hanya volume token:

- Kualitas, keragaman, dan provenance.
- Lisensi, consent, dan privasi.
- Deduplikasi dokumen dan benchmark contamination detection.
- Keseimbangan bahasa, domain, pengetahuan, kode, matematika, dan percakapan.
- Rasio data, parameter, dan compute yang tepat.
- Filtering konten berkualitas rendah dan manipulatif.
- Data reasoning serta data sintetis yang melewati verifier.

Data buruk dalam jumlah besar tetap menghasilkan model besar yang buruk.

### Training recipe

Compute harus diarahkan melalui resep training yang tepat:

- Tokenizer dan vocabulary.
- Arsitektur dan ukuran model.
- Initialization dan parameterization.
- Optimizer dan learning-rate schedule.
- Global batch size dan token budget.
- Precision dan distributed training strategy.
- Regularization dan data curriculum.
- Monitoring instability, loss spike, dan hardware failure.

Training frontier dapat gagal setelah menghabiskan compute besar apabila
stabilitas numerik dan engineering training tidak matang.

### Post-training

Base model yang kuat belum otomatis menjadi asisten yang kuat. Dibutuhkan:

- Supervised fine-tuning.
- Preference optimization.
- Reasoning reinforcement learning.
- Safety dan robustness training.
- Tool-use training.
- Calibration dan uncertainty behavior.

Pretraining menentukan kemampuan yang tersedia. Post-training menentukan kapan
dan bagaimana kemampuan itu digunakan.

### Evaluation

Evaluasi harus membedakan kemajuan nyata dari benchmark gaming:

- Public dan private benchmark.
- Contamination detection.
- Human evaluation.
- Adversarial dan safety testing.
- Regression suite.
- Evaluasi tugas produk nyata.

Evaluation menentukan apakah setiap iterasi bergerak ke arah yang benar.

### Inference dan product system

Frontier assistant bukan hanya checkpoint model:

- System prompt dan policy.
- Tool calling.
- Search dan retrieval.
- Memory dan context management.
- Model routing.
- Safety controls.
- Efficient serving dan observability.

Tools memperluas kemampuan di luar weights. Inference system menentukan berapa
banyak kemampuan model yang benar-benar sampai kepada pengguna.

### Tim dan iteration loop

Eksekusi frontier membutuhkan gabungan kepakaran:

- Machine learning research.
- Distributed systems.
- Data engineering.
- Post-training dan evaluation.
- Inference optimization.
- Security, privacy, dan safety.
- Product engineering.

Hambatan utama setelah compute tersedia berpindah ke data berkualitas,
pengetahuan resep training, reliability, dan feedback loop end-to-end. Membuat
Transformer besar relatif jelas; membuatnya stabil, cerdas, dapat dipercaya,
dan berguna adalah bagian yang paling sulit.

## Data Flywheel dari Usage Records

Usage record dari platform yang menjalankan banyak model dapat menjadi aset
utama untuk post-training, evaluasi, routing, dan observability. Record yang
berguna idealnya menyimpan:

```text
prompt
response
model dan versi
system prompt/policy version
latency
input/output token usage
cost
tool calls dan hasil tool
error/retry
feedback atau rating
task outcome
timestamp dan provenance
```

### Potensi penggunaan

- **Supervised fine-tuning:** mengambil percakapan dan jawaban berkualitas tinggi.
- **Preference optimization:** membandingkan beberapa jawaban untuk prompt yang sama.
- **Distillation:** memindahkan pola kemampuan model teacher besar ke SLM.
- **Model routing:** memilih model berdasarkan tugas, kualitas, latency, dan biaya.
- **Tool-use training:** mempelajari trajectory penggunaan tool yang berhasil.
- **Private evaluation:** membuat benchmark dari kebutuhan pengguna nyata.
- **Reward/quality model:** memperkirakan kualitas respons sebelum dikirim.
- **Failure analysis:** menemukan pola hallucination, refusal salah, error, dan retry.
- **Synthetic data:** memilih seed nyata lalu menghasilkan variasi yang diverifikasi.

Jika satu prompt memiliki beberapa jawaban model, record menjadi sangat bernilai:

```text
prompt
|- model A: benar, mahal, lambat
|- model B: benar, murah, cepat
`- model C: salah
```

Data ini dapat melatih preference model sekaligus router yang mencari model
termurah dan tercepat yang masih memenuhi ambang kualitas.

### Pipeline sebelum training

Raw log tidak boleh langsung dijadikan dataset:

```text
raw usage records
-> consent dan licensing filter
-> PII, credential, dan secret redaction
-> normalization dan deduplication
-> language dan domain classification
-> quality scoring
-> factual/tool verification
-> train/validation/private-evaluation split
-> dataset SFT, preference, routing, atau evaluation
```

Hal yang wajib dijaga:

- Izin pengguna dan dasar legal untuk penggunaan data training.
- Terms provider terkait penggunaan output untuk melatih model lain.
- Penghapusan PII, API key, credential, dan dokumen privat.
- Isolasi tenant dan access control.
- Filtering prompt injection serta konten berbahaya.
- Deduplikasi dan pencegahan benchmark contamination.
- Provenance, audit trail, retention, dan hak penghapusan data.
- Verifikasi agar respons salah tidak dipelajari sebagai jawaban benar.

### Data flywheel produksi

Nilai terbesar usage record adalah loop perbaikan yang terus berjalan:

```text
pengguna memberi tugas
-> model/router menjawab
-> outcome dan feedback direkam
-> kegagalan dikelompokkan
-> data berkualitas disusun dan diverifikasi
-> model/router diperbaiki
-> hasil diuji pada tugas nyata
```

Usage data biasanya terlalu conversational dan sempit untuk menjadi satu-satunya
sumber pretraining base model. Namun, data tersebut sangat kuat untuk membangun
SLM yang unggul pada distribusi tugas nyata milik platform.

## Katalog Dataset

Dataset harus dipilih berdasarkan tahap training. Dataset pretraining, SFT,
preference, reasoning, dan evaluation memiliki tujuan serta format berbeda dan
tidak boleh dicampur tanpa mengetahui objective-nya.

### Dataset yang sudah dipertimbangkan

| Dataset | Tahap | Catatan |
|---|---|---|
| OpenWebText | Pretraining | Replikasi distribusi WebText; kualitas bervariasi |
| FineWeb/FineWeb-Edu | Pretraining | Pilihan web modern; mulai dari sample kecil atau FineWeb-Edu |
| Wikipedia | Pretraining | Relatif bersih dan faktual, tetapi gaya serta domain sempit |
| TinyStories | Pretraining eksperimen | Bagus untuk model kecil dan validasi pipeline |
| The Stack | Pretraining kode | Sangat besar; periksa lisensi, provenance, dan opt-out |
| Alpaca | SFT | Mudah dipelajari tetapi tua dan kualitas sintetisnya tidak merata |
| OpenAssistant 1/2 | SFT/preference | Percakapan multilingual dan struktur message |
| UltraChat | SFT | Percakapan sintetis skala besar |
| UltraFeedback | Preference/DPO | Perbandingan dan penilaian beberapa respons |

### Base pretraining

- **SlimPajama:** campuran domain terfilter untuk eksperimen pretraining.
- **Dolma:** korpus terbuka dengan dokumentasi dan provenance yang relatif baik.
- **RedPajama-V2:** berguna untuk mempelajari filtering web corpus sendiri.
- **C4/mC4:** baseline klasik; perlu perhatian pada kualitas dan deduplikasi.
- **CulturaX:** korpus multilingual yang relevan untuk bahasa Indonesia.
- **OSCAR:** Common Crawl multilingual dengan subset per bahasa.
- **MADLAD-400:** cakupan bahasa yang luas.
- **Cosmopedia:** data textbook dan cerita sintetis untuk pretraining.
- **SmolLM-Corpus:** referensi komposisi data untuk model kecil.

### Bahasa Indonesia dan Nusantara

- **Indo4B:** korpus pretraining bahasa Indonesia.
- **Wikipedia Indonesia:** baseline bersih dan mudah diaudit.
- **CulturaX Indonesian subset:** sumber volume multilingual terfilter.
- **OSCAR Indonesian subset:** sumber web Indonesia yang perlu cleaning.
- **CC-100 Indonesian:** sumber tambahan dengan kebutuhan cleaning berat.
- **IndoNLU:** kumpulan tugas pemahaman bahasa; terutama untuk evaluasi.
- **IndoNLG:** tugas generatif bahasa Indonesia.
- **NusaX/NusaWrites:** bahasa Indonesia dan bahasa lokal Nusantara.
- **OPUS:** parallel corpus untuk translasi dan kemampuan multilingual.
- **SEACrowd:** katalog dan koleksi dataset Asia Tenggara.

### Instruction tuning

- **FLAN Collection:** ragam tugas dan instruksi.
- **Dolly 15K:** baseline SFT kecil dan mudah dipahami.
- **OpenOrca:** data instruction/reasoning sintetis.
- **SlimOrca:** varian yang lebih terkurasi.
- **WizardLM/Evol-Instruct:** instruksi kompleks sintetis.
- **Aya Dataset/Collection:** instruction tuning multilingual.
- **OpenHermes:** campuran instruction/chat yang perlu audit sumber.
- **Tulu mixtures:** referensi komposisi post-training modern.

### Preference dan alignment

- **Anthropic HH-RLHF:** pasangan helpful/harmless.
- **UltraFeedback:** DPO dan reward modeling.
- **HelpSteer:** respons dengan beberapa atribut kualitas.
- **PKU-SafeRLHF:** eksperimen safety alignment.
- **SHP:** preference pairs dari Reddit.
- **OpenAssistant preference pairs:** data conversational multilingual.

### Reasoning dan data terverifikasi

- **GSM8K:** matematika dasar dengan jawaban yang dapat diperiksa.
- **MATH:** soal matematika yang lebih sulit.
- **NuminaMath:** data matematika skala besar.
- **OpenMathInstruct:** instruction matematika sintetis.
- **MetaMathQA:** variasi soal dan solusi matematika.
- **CodeContests/APPS:** coding dengan test sebagai verifier.
- **The Stack/permissive code datasets:** pretraining kode, bukan langsung reasoning.

### Urutan eksperimen

Urutan yang menjaga fokus dari pipeline kecil menuju model Indonesia:

```text
1. TinyStories untuk membuktikan tokenizer dan training pipeline
2. Wikipedia Indonesia untuk baseline bahasa yang relatif bersih
3. FineWeb-Edu sample + CulturaX/OSCAR Indonesia
4. Campuran Indonesia + English + code + matematika
5. Aya/OpenAssistant/UltraChat untuk SFT
6. UltraFeedback atau preference data internal untuk DPO
7. Usage records terverifikasi untuk specialization dan routing
```

Baseline komposisi awal yang dapat diuji, bukan resep final:

```text
60% bahasa Indonesia
20% English berkualitas/edukatif
10% code
10% matematika, dialog, dan instruction
```

Komposisi harus ditentukan ulang berdasarkan target kemampuan, hasil evaluasi,
dan efisiensi tokenizer pada setiap bahasa/domain.

### Checklist sebelum digunakan

- Lisensi dataset dan dokumen sumber.
- Izin penggunaan komersial dan training model turunan.
- Provenance dan mekanisme opt-out.
- PII, credential, rahasia, dan data sensitif.
- Normalisasi format dan encoding.
- Deduplikasi di dalam dan antar-dataset.
- Benchmark contamination.
- Proporsi machine-generated text.
- Kualitas bahasa Indonesia dan code-switching.
- Kebenaran faktual dan kualitas solusi.
- Format prompt, response, message, preference, atau trajectory.
- Pemisahan train, validation, private evaluation, dan holdout temporal.

Untuk tahap awal, jangan mulai dari FineWeb atau The Stack penuh. Sample kecil
yang terukur lebih berguna untuk memahami data pipeline, training dynamics, dan
evaluation sebelum skala data memperbesar biaya setiap kesalahan.

## Strategi Hardware dan Eksperimen Satu Jam

Proses belajar ini perlu dijaga tetap menyenangkan dan menghasilkan pengetahuan.
Prinsip kerja setiap sesi:

> Satu eksperimen berdurasi maksimal sekitar satu jam harus menjawab satu
> pertanyaan yang terukur.

Contoh pertanyaan:

- Apakah tokenizer BPE 8k lebih efisien daripada 16k untuk bahasa Indonesia?
- Apakah SwiGLU menurunkan validation loss dibanding ReLU?
- Apakah RoPE lebih baik daripada learned position embedding?
- Berapa campuran data Indonesia dan Inggris yang paling efektif?
- Apakah model 30M mengalahkan 10M pada token budget yang sama?

### Pembagian hardware

| Hardware | Kekuatan | Penggunaan utama |
|---|---|---|
| RTX 4060 8 GB | Iterasi lokal cepat | Development, debugging, profiling, training 1M-50M |
| Colab T4 16 GB | VRAM lebih besar dan remote | Eksperimen 20M-100M, portability, LoRA/QLoRA |
| DGX Spark 128 GB unified | Kapasitas memori besar | 100M-500M, long context, continued pretraining, distillation |

DGX Spark terutama membuka kapasitas model dan context yang lebih besar. Ia
tidak otomatis lebih cepat daripada seluruh GPU datacenter, sehingga throughput
nyata tetap perlu diukur.

### RTX 4060

Target eksperimen yang nyaman:

```text
1M-30M parameter
context 128-512
10M-100M training token
10-60 menit per eksperimen
```

Gunakan untuk tokenizer, implementasi model, data pipeline, ablation kecil,
profiling, generation, dan Nano Gatra. Jangan menghabiskan waktu hardware yang
lebih mahal untuk mencari typo, shape mismatch, atau bug dasar.

### Colab T4

Gunakan T4 untuk batch/context yang tidak muat di RTX 4060, validasi bahwa
project reproducible di mesin lain, dan menjalankan training saat laptop sedang
dipakai. Target praktis:

```text
20M-100M dari nol untuk eksperimen terbatas
0.5B-3B untuk LoRA/QLoRA
```

Karena sesi Colab dapat terputus, checkpoint periodik, resume otomatis, dan
sinkronisasi metrics/artefak ke persistent storage adalah fitur wajib.

### DGX Spark

Gunakan setelah pipeline lolos smoke test dan study run lokal:

- Gatra 100M-500M dari nol.
- Context lebih panjang dan batch lebih besar.
- Continued pretraining model 0.5B-3B.
- LoRA/QLoRA model yang lebih besar.
- Teacher inference untuk synthetic data.
- Distillation.
- Batch evaluation dengan model judge dan verifier.

Model 1B+ dari nol tetap membutuhkan token budget dan durasi besar agar benar-
benar bagus. Dalam sesi satu jam, model 100M-350M lebih cocok untuk eksperimen
scaling daripada target produk final.

## Kapan Model Mulai Terasa Nyata

### 1M-10M parameter

- Belajar ejaan, format, dan pola lokal.
- Cocok untuk validasi implementasi.
- Belum menjadi asisten yang dapat dipercaya.

### 30M-100M parameter

- Mulai menghasilkan paragraf sederhana.
- Dapat mengikuti pola instruksi yang terbatas.
- Dapat menunjukkan generalisasi pada domain sempit.
- Titik menarik untuk model from-scratch yang mulai terasa nyata.

### 100M-500M parameter

- Base model kecil yang berguna untuk riset efisiensi dan specialization.
- Bahasa lebih konsisten jika data dan token budget memadai.
- Instruction following sederhana setelah SFT.
- Target jangka menengah Gatra from scratch.

### Pretrained 0.5B-3B dengan post-training

- Jalur tercepat menuju percakapan dan instruction following yang berguna.
- Cocok untuk RAG, tool calling, domain specialization, dan deployment lokal.
- Tidak perlu mengulang seluruh biaya pretraining dari nol.

## Dua Track Gatra

### Track A: Gatra from scratch

Tujuan utamanya riset, pemahaman, dan pengukuran scaling:

```text
Gatra-1M
-> Gatra-10M
-> Gatra-30M
-> Gatra-100M
-> Gatra-350M
```

Semua ukuran sebaiknya menggunakan pipeline, tokenizer, benchmark, dan format
laporan yang konsisten agar hasilnya dapat dibandingkan.

### Track B: Gatra yang berguna

Tujuan utamanya produk dan kemampuan praktis:

```text
pretrained base 0.5B-3B
-> continued pretraining Indonesia/English
-> supervised instruction tuning
-> preference optimization
-> tool calling dan retrieval
-> quantization dan deployment
```

Track A membangun kompetensi membuat model. Track B memberikan model berguna
lebih cepat. Keduanya saling memberi data, evaluator, dan pelajaran engineering.

## Mode Training

Sediakan tiga profil agar eksperimen tidak selalu berubah menjadi training
panjang:

```text
smoke : 1-5 menit, memastikan pipeline benar
study : 30-60 menit, menjawab satu hipotesis
run   : beberapa jam atau semalam, kandidat milestone
```

Perbandingan eksperimen sebaiknya memakai jumlah token yang telah diproses:

```text
tokens_seen = batch_size x block_size x optimizer_steps
```

Iteration count saja tidak adil ketika batch size, context, gradient
accumulation, atau hardware berbeda.

## Otomasi Eksperimen

Setiap training run idealnya otomatis:

1. Membaca file konfigurasi.
2. Menyimpan snapshot config dan commit hash.
3. Menghitung jumlah parameter dan token budget.
4. Mencatat loss, perplexity, token per detik, dan penggunaan VRAM.
5. Mengevaluasi prompt tetap dan benchmark kecil.
6. Menyimpan latest dan best checkpoint.
7. Mendukung resume tanpa mengulang data secara keliru.
8. Berhenti rapi berdasarkan batas waktu.
9. Menghasilkan laporan perbandingan dengan baseline.

Target antarmuka:

```text
uv run train.py --config configs/gatra-30m.toml --max-time 60m
```

Dengan time budget, training dapat dibiarkan berjalan sambil mengerjakan hal
lain dan tetap menghasilkan checkpoint serta laporan yang valid.

## Milestone Praktis

1. Pisahkan dataset pretraining dan SFT.
2. Latih tokenizer BPE Indonesia 8k/16k.
3. Modernisasi model dengan RoPE, RMSNorm, SwiGLU, dan weight tying.
4. Tambahkan config, checkpoint, resume, mixed precision, dan time limit.
5. Tetapkan Gatra-10M sebagai baseline pertama.
6. Latih Gatra-30M dan bandingkan pada token budget yang sama.
7. Jalankan Gatra-100M di T4 atau DGX Spark.
8. Adaptasi pretrained 0.5B-3B menjadi Gatra Instruct.
9. Gunakan usage records untuk evaluation, preference, dan routing.
10. Turunkan Gatra menjadi Tecton melalui code continued pretraining dan
    execution-verified SFT/RL.

Target awal yang seimbang:

> Gatra-30M dari nol sebagai model riset, lalu Gatra-1B-Instruct dari pretrained
> base sebagai model yang berguna.
