# Face Mask Detection

Bu proje, derin öğrenme kullanarak yüz görüntülerinden maskeli veya maskesiz olduğunu tespit eden bir sistemdir. Model PyTorch ile eğitilmiş ve Gradio arayüzü ile Hugging Face Spaces üzerinden çalıştırılmaktadır.

## Kullanım

1. Hugging Face linkinden uygulamayı açın:
   [Public Space Link](https://huggingface.co/spaces/kadirkartal/face-mask-docker)
2. Yüz resmi yükleyin.
3. Sistem "Maskeli" veya "Maskesiz" olarak tahmin edecektir.

## Dosyalar

- `app.py` → Gradio arayüzü
- `Dockerfile` → HF Space deploy
- `requirements.txt` → Paket listesi
- `mask_model.pth` → Eğitilmiş CNN modeli
