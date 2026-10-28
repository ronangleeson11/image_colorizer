import pickle
import tarfile
import matplotlib.pyplot as plt

# Open dataset
with tarfile.open("cifar-100-python.tar.gz", "r:gz") as tar:
    meta_file = tar.extractfile("cifar-100-python/meta")
    train_file = tar.extractfile("cifar-100-python/train")
    meta_data = pickle.load(meta_file, encoding="latin1")
    train_data = pickle.load(train_file, encoding="latin1")

# Reshape images
raw_images = train_data["data"]  # 50000 (images) x 3072 (pixels)
images = raw_images.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)  # 50000 (images) x 32 (pixels) x 32 (pixels) x 3 (RGB)

# Plot some images
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
axes = axes.ravel()

for i in range(9):
    axes[i].imshow(images[i])  # Display image
    axes[i].axis("off")  # Hide axes

plt.tight_layout()
plt.show()
