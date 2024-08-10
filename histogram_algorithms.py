import numpy as np
import matplotlib.pyplot as plt

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
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

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

