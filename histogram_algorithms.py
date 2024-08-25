import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def histogram_equalization(image, L=256, scale=255):
    hist, _ = np.histogram(image.flatten(), L, [0, L])
    total_pixels = image.flatten().size
    norm_hist = hist / total_pixels
    cdf = np.cumsum(norm_hist)
    equalized_transform = np.floor(cdf * scale).astype(int)
    equalized_image = equalized_transform[image]
    
    return equalized_image

def cumulative_histogram_equalization(image, L=256, scale=255):
    hist, _ = np.histogram(image.flatten(), L, [0, L])
    total_pixels = image.flatten().size
    norm_hist = hist / total_pixels
    cdf = np.cumsum(norm_hist)
    cdf_normalized = np.floor((cdf - cdf.min()) / (cdf.max() - cdf.min()) * scale).astype(int)
    return cdf_normalized[image]

def qdhe(image, L=256):
    def calculate_histogram(image, L=256):
        hist, _ = np.histogram(image.flatten(), L, [0, L])
        return hist

    def cumulative_distribution_function(hist):
        total_pixels = hist.sum()   
        if total_pixels == 0:
            raise ValueError("The total number of pixels in the histogram is zero. The input image might be empty or invalid.")
        norm_hist = hist / total_pixels
        cdf = np.cumsum(norm_hist)
        return cdf

    def partition_histogram(image, hist):
        total_pixels = hist.sum()
        cdf = cumulative_distribution_function(hist)

        # Find intensity values corresponding to these median indices
        m0 = image.flatten().min()
        m1 = np.searchsorted(cdf, 0.25) 
        m2 = np.searchsorted(cdf, 0.5) 
        m3 = np.searchsorted(cdf, 0.75) 
        m4 = image.flatten().max() 

        return [m0, m1, m2, m3, m4]

    def clip_histogram(hist, threshold):
        clipped_hist = np.minimum(hist, threshold)
        return clipped_hist

    def allocate_gray_levels(hist, m, L=256):
        total_pixels = hist.sum()
        
        spans = [m[i + 1] - m[i] for i in range(len(m) - 1)]
        total_span = sum(spans)
        ranges = [(L - 1) * span / total_span for span in spans]
        
        starts = [m[0]]  # Initial start value
        ends = []

        for i in range(len(ranges)):
            if i > 0:
                starts.append(ends[i - 1] + 1)  # Calculate start for current interval
            end = starts[i] + ranges[i]  # Calculate end for current interval
            ends.append(end)

        return starts, ends    

    def histogram_equalization(cdf, start, end):
        equalized_hist = cdf * (end - start) + start
        return equalized_hist
        
    #Calculate histogram
    hist = calculate_histogram(image)
    
    # Histogram partitioning
    m = partition_histogram(image, hist)

    # Clipping
    threshold = hist.mean()
    clipped_hist = clip_histogram(hist, threshold)

    # Allocate gray levels
    starts, ends = allocate_gray_levels(clipped_hist, m, L)
    equalized_image = np.zeros_like(image)

    for i, (start, end) in enumerate(zip(starts, ends)):
        start = int(start)  # Convert to integer
        end = int(end)  # Convert to integer
        
        mask = (image >= start) & (image <= end)
        
        # Extract the sub-histogram for the current intensity range
        sub_hist = clipped_hist[start:end + 1]
        
        # Calculate the cumulative distribution function (CDF) for the sub-histogram
        cdf = cumulative_distribution_function(sub_hist)
        
        # Apply histogram equalization to the sub-histogram
        equalized_hist = histogram_equalization(cdf, start, end)
        
        # Map the original pixel values to the equalized values
        equalized_image[mask] = equalized_hist[image[mask] - start]  # Offset by start

    return equalized_image

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply techniques
he_image = histogram_equalization(image)
che_image = cumulative_histogram_equalization(image)
qdhe_image = qdhe(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Display images
plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 3, 2), plt.imshow(he_image, cmap='gray'), plt.title('Histogram Equalized')
plt.subplot(2, 3, 3), plt.imshow(che_image, cmap='gray'), plt.title('Cumulative Histogram Equalized')
plt.subplot(2, 3, 4), plt.imshow(qdhe_image, cmap='gray'), plt.title('Quadrant Dynamic Histogram Equalized')
plt.subplot(2, 3, 5), plt.imshow(clahe_image, cmap='gray'), plt.title('CLAHE')
plt.tight_layout()
plt.show()

# Define metrics calculation functions
def calculate_mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calculate_psnr(mse, L=256):
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(((L-1) ** 2) / mse)
    return psnr

def calculate_sd(image):
    sd = np.std(image)
    return sd

def process_images_in_dataset(dataset_path):
    mse_results = {'HE': [], 'CHE': [], 'QDHE': [], 'CLAHE': []}
    psnr_results = {'HE': [], 'CHE': [], 'QDHE': [], 'CLAHE': []}
    sd_results = {'HE': [], 'CHE': [], 'QDHE': [], 'CLAHE': [], 'Original': []}

    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply histogram methods
            
            try:
                he_image = histogram_equalization(image)
                che_image = cumulative_histogram_equalization(image)
                qdhe_image = qdhe(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe.apply(image)

                # Calculate metrics
                for method, processed_image in zip(['HE', 'CHE', 'QDHE', 'CLAHE'], [he_image, che_image, qdhe_image, clahe_image]):
                    mse = calculate_mse(image, processed_image)
                    psnr = calculate_psnr(mse)
                    sd = calculate_sd(processed_image)
                
                    mse_results[method].append(mse)
                    psnr_results[method].append(psnr)
                    sd_results[method].append(sd)
            
                # Calculate SD for the original image
                sd_results['Original'].append(calculate_sd(image))

            except:
                continue       
        
    # Aggregate results
    mean_mse = {method: np.mean(mse) for method, mse in mse_results.items()}
    mean_psnr = {method: np.mean(psnr) for method, psnr in psnr_results.items()}
    mean_sd = {method: np.mean(sd) for method, sd in sd_results.items()}
    
    return mean_mse, mean_psnr, mean_sd

# Define path to your dataset
dataset_path = 'images/landscape Images/gray/'

# Process dataset and get metrics
mean_mse, mean_psnr, mean_sd = process_images_in_dataset(dataset_path)

# Prepare data for plotting
methods = ['Original', 'HE', 'CHE', 'QDHE', 'CLAHE']
mse_values = [mean_mse[method] for method in ['HE', 'CHE', 'QDHE', 'CLAHE']]
psnr_values = [mean_psnr[method] for method in ['HE', 'CHE', 'QDHE', 'CLAHE']]
sd_values = [mean_sd['Original']] + [mean_sd[method] for method in ['HE', 'CHE', 'QDHE', 'CLAHE']]

# Plot metrics
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# MSE plot
axs[0].bar(['HE', 'CHE', 'QDHE', 'CLAHE'], mse_values, color = ['royalblue', 'tomato', 'mediumseagreen', 'darkorange'], edgecolor='black')
for bar in axs[0].patches:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
axs[0].set_xlabel('Histogram Methods')
axs[0].set_ylabel('Mean Square Error (MSE)')
axs[0].set_title('MSE Comparison')
axs[0].set_ylim(0, max(mse_values) * 1.1)  # Adjust y-axis limit dynamically
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# PSNR plot
axs[1].bar(['HE', 'CHE', 'QDHE', 'CLAHE'], psnr_values, color=['royalblue', 'tomato', 'mediumseagreen', 'darkorange'], edgecolor='black')
for bar in axs[1].patches:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
axs[1].set_xlabel('Histogram Methods')
axs[1].set_ylabel('Peak Signal to Noise Ratio (PSNR)')
axs[1].set_title('PSNR Comparison')
axs[1].set_ylim(0, max(psnr_values) * 1.1) 
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# SD plot
axs[2].bar(methods, sd_values, color=['slateblue','royalblue', 'tomato', 'mediumseagreen', 'darkorange'], edgecolor='black')
for bar in axs[2].patches:
    height = bar.get_height()
    axs[2].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
axs[2].set_xlabel('Histogram Methods')
axs[2].set_ylabel('Standard Deviation (SD)')
axs[2].set_title('SD Comparison')
axs[2].set_ylim(0, max(sd_values) * 1.1) 
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
