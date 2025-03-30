import urllib.request
from PIL import Image
import numpy as np
from PIL import Image
from scipy import ndimage
import skimage.io
from skimage import exposure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
#URL and Image NAME examples: https://assets.science.nasa.gov/dynamicimage/assets/science/psd/mars/resources/detail_files/6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg?w=2560&format=webp&fit=clip&crop=faces%2Cfocalpoint', "6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg
urllib.request.urlretrieve(
'https://assets.science.nasa.gov/dynamicimage/assets/science/psd/mars/resources/detail_files/6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg?w=2560&format=webp&fit=clip&crop=faces%2Cfocalpoint',
"6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg")
#image name
#img = Image.open("6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg")
#print("HELLO")
# doesn't work img.show()
#display(img)
#print("Original")
# Load the image using skimage
img = skimage.io.imread("6944_Mars-Opportunity-blueberries-hematite-Fram-Crater-pia19113-full2.jpg")
# Enhance contrast using Histogram Equalization
img_equalized = exposure.equalize_hist(img)
# Enhance contrast using Histogram Equalization
img_equalized = exposure.equalize_hist(img)
# Enhance contrast using Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
# Enhance sharpness using Unsharp Masking
img_unsharp = skimage.filters.unsharp_mask(img, radius=3, amount=2)
# Convert data type to uint8 for better compatibility
img_equalized = (img_equalized * 255).astype(np.uint8)
img_adapteq = (img_adapteq * 255).astype(np.uint8)
img_unsharp = (img_unsharp * 255).astype(np.uint8)
# Create subplots for better visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# Display the images with titles
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(img_equalized)
axes[0, 1].set_title("Histogram Equalized")
axes[1, 0].imshow(img_adapteq)
axes[1, 0].set_title("Adaptive Equalized")
axes[1, 1].imshow(img_unsharp)
axes[1, 1].set_title("Unsharp Masked")
# Turn off axis ticks for cleaner presentation
for ax in axes.flat:
ax.axis('off')
# Adjust spacing between subplots (
plt.tight_layout()
# This codes shows the the plot (images)
plt.show()
# Calculates PSNR and SSIM
psnr_O = compare_psnr(img, img)
ssim_O = compare_ssim(img, img, multichannel=True, win_size=3) # Changed win_size to 3
psnr_equalized = compare_psnr(img, img_equalized)
ssim_equalized = compare_ssim(img, img_equalized, multichannel=True, win_size=3) # Changed win_size to 3
psnr_adapteq = compare_psnr(img, img_adapteq)
ssim_adapteq = compare_ssim(img, img_adapteq, multichannel=True, win_size=3) # Changed win_size to 3
psnr_unsharp = compare_psnr(img, img_unsharp)
ssim_unsharp = compare_ssim(img, img_unsharp, multichannel=True, win_size=3) # Changed win_size to 3
# Print the results
print(f"Original Equalization: PSNR = {psnr_O:.2f}, SSIM = {ssim_O:.4f}")
print(f"Histogram Equalization: PSNR = {psnr_equalized:.2f}, SSIM = {ssim_equalized:.4f}")
print(f"Adaptive Equalization: PSNR = {psnr_adapteq:.2f}, SSIM = {ssim_adapteq:.4f}")
print(f"Unsharp Masking: PSNR = {psnr_unsharp:.2f}, SSIM = {ssim_unsharp:.4f}")
