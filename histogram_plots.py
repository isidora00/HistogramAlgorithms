import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images_and_histograms(images, titles = None,xticks=None, labels=None, y_max=6000):
    n = len(images)
    fig, ax = plt.subplots(2, n)

    for i, image in enumerate(images):
        ax[0, i].imshow(image, cmap='gray', vmin=0, vmax=256)
        ax[0, i].axis('off')
        if titles:
            ax[0,i].set_title(titles[i])
        
        hist, _ = np.histogram(image, 256, [0, 256])
        ax[1, i].plot(range(256), hist, color='black', alpha = 0.5)
        ax[1, i].fill_between(range(256), hist, color='black', alpha = 0.3)
        ax[1, i].set_xlim([0, 255])
        ax[1, i].set_ylim([0, y_max])
        ax[1, i].set_yticklabels([])
        ax[1, i].set_xlabel("intensity")
        ax[1, i].set_ylabel("frequency")
        if xticks:
            ax[1, i].set_xticks(ticks = xticks[i], labels=labels[i])
            ax[1, i].grid(True, axis='x',color = "black",linestyle='--',linewidth=2)
                

    plt.tight_layout()
    plt.show()

    
def show_histogram(hist, xticks=None, labels=[r"$I_{min}$", r"$Q_{1}$", r"$Q_{2}$", r"$Q_{3}$", r"$I_{max}$"]): 
    plt.plot(range(256), hist, color='black', alpha = 0.5)
    plt.fill_between(range(256), hist,color='black', alpha = 0.3)
    if xticks:
        plt.xticks(ticks = xticks, labels=labels)
        plt.grid(True, axis='x',color = "black",linestyle='--',linewidth=2)

    plt.xlim([0,255])
    plt.ylim([0,4000])
    plt.show()
    
def show_clipped_histogram(hist, clipped_hist, xticks=None, labels=[r"$I_{min}$", r"$Q_{1}$", r"$Q_{2}$", r"$Q_{3}$", r"$I_{max}$"]):
    plt.plot(range(256), clipped_hist, color = 'black', alpha = 0.7)
    plt.fill_between(range(256), hist, color = 'black', alpha = 0.3)
    if xticks:
        plt.xticks(ticks = xticks, labels=labels)
        plt.grid(True, axis='x',color = "black",linestyle='--',linewidth=2)
    plt.ylim([0,4000])
    plt.xlim([0,256])
    plt.show()