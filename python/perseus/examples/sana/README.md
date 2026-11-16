# Encrypted text to image

Gemma-2-2B and SANA-Sprint 0.6B turn a text prompt into a 768 × 768 image fully encrypted, in 269 s on one
GPU. Both images below were generated this way.

<p align="center">
  <img src="../../../../assets/t2i-cat.jpg" width="49%" alt="An encrypted SANA-Sprint image for the prompt a photo of a cat, a tabby kitten looking at the camera.">
  <img src="../../../../assets/t2i-landscape.jpg" width="49%" alt="An encrypted SANA-Sprint image for the prompt a landscape at the sunset, a meadow under an orange and purple sky.">
</p>

```bash
perseus-t2i "a photo of a cat"
perseus-t2i "a landscape at the sunset"
```
