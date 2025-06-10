#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam


np.random.seed(2)


Origional_data=pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx",sheet_name='16+3',header=0)
Per_data=Origional_data.iloc[0:16, :]
df=Per_data[['lg(O3)','lg(H2O2)','pH','TOC','FRI','OU120']]

input_dim = df.shape[1]

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(Dense(256))
    model.add(Dense(512))
    model.add(Dense(input_dim, activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

latent_dim = 64
batch_size = 6
epochs = 500

generator = build_generator(latent_dim=latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = scaler.fit_transform(df)

d_losses = []
g_losses = []
for epoch in range(epochs):
    real = df_scaled[np.random.randint(0, df_scaled.shape[0], batch_size)]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_losses.append(d_loss)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    g_losses.append(g_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}')

noise = np.random.normal(0, 1, (36, latent_dim))
generated_data = generator.predict(noise)

generated_data = scaler.inverse_transform(generated_data)

print("Generated 50 rows of data:")
print(generated_data)

plt.rcParams['figure.dpi'] = 300

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

plt.figure(figsize=(6, 4))
plt.plot(d_losses, label='D Loss')
plt.plot(g_losses, label='G Loss')
plt.title('GAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

df_toc = df['TOC'].values
gen_toc = generated_data[:, list(df.columns).index('TOC')]
plt.figure(figsize=(6, 4))
plt.violinplot([df_toc, gen_toc])
plt.title('Distribution of TOC in Original and Generated Data')
plt.xticks(ticks=[1, 2], labels=['Original Data', 'Generated Data'])
plt.ylabel('TOC Value')
plt.show()

