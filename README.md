# Proyek Submission Belajar Fundamental Generative AI (BFGAI)

Submission ini mengharuskan untuk melalui dua tahapan, yaitu eksperimen (Pipeline) dan pembuatan antarmuka (Streamlit).

## Perhatian Penting
1. Pastikan notebook dapat dijalankan sepenuhnya tanpa error sebelum dikirimkan.
2. Penuhi terlebih dahulu kriteria **Basic** sebelum ke **Skilled** dan **Advanced**.
3. Jika mengalami keterbatasan komputasi, disarankan memakai GPU free tier di Google Colab atau Kaggle.
4. Bebas menggunakan prompt apa pun (tanpa unsur SARA/negatif). Sangat disarankan menggunakan prompt yang sama untuk perbandingan.
5. Dilarang menggunakan model SDXL karena rawan OoM (Out-of-Memory). Gunakan `runwayml/stable-diffusion-v1-5`.
6. Hindari memuat ulang model berulang kali. Gunakan instance model yang sama untuk efisiensi VRAM.

---

## Kriteria 1: Melakukan Image Generation dari Teks (Text-to-Image)

**Basic (2 pts)**
- Membuat fungsi `generate_simple_image()` menggunakan Stable Diffusion Pipeline (`runwayml/stable-diffusion-v1-5`) dengan parameter dasar: Prompt, Negative_prompt, Seed.
- Membuat fungsi `generate_advanced_image()` dengan parameter tambahan: Guidance_scale, num_inference_step.
- Generate gambar semirip mungkin dengan contoh di template menggunakan *Seed=222* dan *negative prompt="photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration"*.
- Pastikan prompt pada kedua fungsi sama.

**Skilled (3 pts)**
- Semua ketentuan Basic terpenuhi.
- Melakukan eksperimen dengan beberapa nilai *Guidance Scale* berbeda dan menuliskan observasinya.
- Melakukan eksperimen jumlah *inference steps* (Rendah: 5-15, Tinggi: 30-50) dan menuliskan perbandingannya.

**Advanced (4 pts)**
- Semua ketentuan Skilled terpenuhi.
- Melakukan *batch inference* (4 gambar sekaligus) dan menampilkannya dalam Grid 2x2.
- Implementasi fungsi `load_scheduler(pipe, scheduler_name)` (Euler A, DPM++, DDIM).
- Menuliskan perbandingan gambar dari 3 Scheduler tersebut.

---

## Kriteria 2: Menyempurnakan Gambar Melalui Image-to-Image

**Basic (2 pts)**
- Membuat fungsi `inpaint_engine(image, mask, prompt)` dengan model `runwayml/stable-diffusion-inpainting`.
- Proses masking dilakukan secara manual (hardcode) via trial & error.
- Hasil inpainting semirip mungkin dengan gambar yang diharapkan (Gunakan Seed=9).

**Skilled (3 pts)**
- Semua ketentuan Basic terpenuhi.
- Menggunakan *model segmentation* untuk menghasilkan mask otomatis.
- Membuat fungsi `prepare_outpainting()` untuk memperluas kanvas ke satu arah.
- Melakukan *outpainting* pada satu sisi menggunakan hasil inpainting.

**Advanced (4 pts)**
- Semua ketentuan Skilled terpenuhi.
- Logika outpainting untuk fitur "Zoom Out" (perluas bertahap ke berbagai arah).
- Menerapkan *Refiner Pattern Logic* (Two-Stage Generation):
  - Stage 1: Pipeline Base (`denoising_end=0.8`).
  - Stage 2: Pipeline Img2Img (`denoising_start=0.8`).

---

## Kriteria 3: Membuat Interface dengan Streamlit

**Basic (2 pts)**
- Melengkapi logic Streamlit pada template.
- Wajib memiliki komponen:
  - Text input (Prompt & Negative prompt).
  - Slider (guidance_scale, num_inference_steps).
  - Tombol Generate.
- Gambar hasil generation wajib tampil setelah proses selesai.
- *Screen record* aplikasi (1-5 menit) disimpan sebagai `.mp4`.

**Skilled (3 pts)**
- Semua ketentuan Basic terpenuhi.
- Input `num_images` untuk batch generation (Grid 2x2).
- Dropdown (Selectbox) untuk memilih Scheduler (Euler A, DPM++, DDIM).
- Tombol "Clear Memory" (`gc.collect()`, `torch.cuda.empty_cache()`).

**Advanced (4 pts)**
- Semua ketentuan Skilled terpenuhi.
- Tab baru khusus Inpainting dan Outpainting (Zoom-out).
- Mengintegrasikan `streamlit-drawable-canvas` untuk menggambar mask langsung di browser.

---

## Struktur Berkas Submission

```
BFGAI_Nama-siswa.zip
├── Pipeline_submission_BFGAI_Nama-siswa.ipynb
├── Streamlit_submission_BFGAI_Nama-siswa.ipynb
├── video_demo_aplikasi_BFGAI.mp4
├── requirements.txt
```

*(Perhatian: File .ipynb harus sudah dijalankan terlebih dahulu sehingga output-nya tersimpan!)*