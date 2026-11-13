"""SANA-Sprint 0.6B text-to-image on ``perseus.impl``: the Gemma-2-2B text encoder and one consistency step of the
SANA transformer under CKKS. The client tokenizes and encrypts the prompt embeddings and the latent noise, decrypts
the 32 x 32 x 32 output latent and decodes it with DC-AE in the clear.
"""
