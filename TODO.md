
# To try
- Use much smaller latent space vector. This guy used 16 and that was suffucient because of how small pokemon datasets are.
    - https://towardsdatascience.com/i-generated-thousands-of-new-pokemon-using-ai-f8f09dc6477e
- Add small translations to the data augmentation
- Explicitly set training=True in the discriminator's BatchNorm layers. That might be able to explain why it sucks in the disc since the disc is toggling between training on/off and the generator isn't.
