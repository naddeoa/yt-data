
# To try
- Label custom categories and make the discriminator predict them as well, include them in the loss
- DiffAugment https://arxiv.org/pdf/2006.10738.pdf
- SpectralNorm with BCE, and then WGAN https://arxiv.org/pdf/1802.05957.pdf
- Use Adamw to save memory
- Use much smaller latent space vector. This guy used 16 and that was suffucient because of how small pokemon datasets are.
    - https://towardsdatascience.com/i-generated-thousands-of-new-pokemon-using-ai-f8f09dc6477e

- Grad-CAM on the discriminator to see what it thinks is important.
- Reshape the latent vector directly to 4x4xsomething instead of the dense layer
