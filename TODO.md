# Diffusion

## Questions
- Should I clip the generated noise to -1,1? And what about using tanh on the model?
- How does attention identify things? Where should it go?
- Why does my denoising process, on some epochs, result in very high values for the noised images towards the end of the noising process? Well outside of the normal distribution. The resulting images are often over exposed or mostly a single color.

# Gans
## To try
- Add new conv and transfer the weights of prev models to prog grow to 256x256
- restart training but use the trained disciminator next time

- Add dense layers at the end of the convolutions for global view
- Skip connections in the deeper models for the strdie=1 layers
- R1 penalty
- Replay buffer
- Hinge loss instead of BCE/EM
- SpectralNorm with BCE, and then WGAN https://arxiv.org/pdf/1802.05957.pdf
- initializer = RandomNormal(mean=0., stddev=0.01), Conv2d(kernel_initializer=initializer)
- consistency regularizatoin
- Multiclass discriminator - Label custom categories and make the discriminator predict them as well, include them in the loss
- BERT to condition on the pokemon descriptions
- Use much smaller latent space vector. This guy used 16 and that was suffucient because of how small pokemon datasets are.
    - https://towardsdatascience.com/i-generated-thousands-of-new-pokemon-using-ai-f8f09dc6477e
- Grad-CAM on the discriminator to see what it thinks is important.
- Reshape the latent vector directly to 4x4xsomething instead of the dense layer

- DONE Use Adamw to save memory
    - No obvious difference
- DONE DiffAugment https://arxiv.org/pdf/2006.10738.pdf
    - Seems a little better than just doing data augmentation outside of the model
- DONE Inception score
    - in the evaluation notebook
- DONE get rid off all non-conv
    - Just used kernel=4,stride=1 to go from original latent space to higher dimensions https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb

## Questions
- is gelu 1-Lipschitz? It appears to be but I don't feel confident yet.