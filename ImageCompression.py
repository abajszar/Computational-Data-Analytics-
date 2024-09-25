#Image Compression through k-means and k-medoids 


import numpy as np
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import sys
import os
from scipy.sparse import csc_matrix

def compute_dist(X,medoids):
    N,D = X.shape
    K = medoids.shape[0]
    return np.sqrt(np.sum((np.reshape(X,[N,1,D])-np.reshape(medoids,[1,K,D]))**2,axis=2))

def compute_manhattan_dist(pixels, medoids):
    return np.sum(np.abs(pixels[:, np.newaxis] - medoids), axis=2)

def mykmeans(pixels, K):
    # Reshape pixels into a 2-D array, with data points corresponding to R,G,B values 
    pixels = pixels.reshape(-1, 3)
    # Total number of pixels 
    m = pixels.shape[0]
    # Randomly initialize cluster centers with existing data points 
    c = pixels[np.random.choice(pixels.shape[0], K, replace=False)]

    
    #Iterate until convergence 
    iterations = 200
    for i in range (iterations):
        # Use compute_dist() function to calculate distance between each data point and each cluster center 
        distance = compute_dist(pixels,c)
        #Assign data points to closest cluster
        assignment = np.argmin(distance, axis = 1)
        
        #Check for empty clusters 
        assigned_clusters = np.unique(assignment)
        empty_clusters = set(range(K)) - set(assigned_clusters)
        
        if empty_clusters:
            K -= len(empty_clusters)
            c = pixels[np.random.choice(pixels.shape[0], K, replace=False)]
            continue  
        
        
        # Update the center for each cluster 
        P = csc_matrix((np.ones(m), (np.arange(m), assignment)), shape=(m, K))
        count = P.sum(axis=0).A1
        c_updated = (P.T.dot(pixels) / count[:, None])
        
        # Break occurs during convergence 
        if np.allclose(c_updated, c): #default tolerance is 0.00001
            break
        else: 
            c = c_updated
            
    return assignment, c

    
def mykmedoids(pixels, K):
    # Reshape pixels into a 2-D array with data points corresponding to R,G,B
    pixels = pixels.reshape(-1, 3)
    m = pixels.shape[0]  # Pixel number

    # Randomly initialize K medoids based on existing pixels
    medoids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]
    
    iterations = 10  

    # Loop through each iteration 
    for i in range(iterations):
        
        # Step 1: Compute Manhattan distances to assign points to medoids 
        distances = compute_manhattan_dist(pixels, medoids)
        assignment = np.argmin(distances, axis=1)
        
         #Check for empty clusters 
        assigned_clusters = np.unique(assignment)
        empty_clusters = set(range(K)) - set(assigned_clusters)

        if empty_clusters:
            K -= len(empty_clusters)
            medoids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]
            continue 
        
        swap_occurred = False
        old_medoids = np.copy(medoids)

        # Step 2: Swap each medoid with the points in its cluster and calculate distortion
        # Loop through each medoid 
        for med in range(K):
            cluster_points = pixels[assignment == med]  # Points assigned to the current medoid
            best_medoid = medoids[med]
            og_distortion = np.sum(compute_manhattan_dist(cluster_points, best_medoid.reshape(1, 3)))
            
            # Swap the medoid with each point in its cluster and check distortion
            for point in cluster_points:
                new_distortion = np.sum(compute_manhattan_dist(cluster_points, point.reshape(1, 3)))
                if new_distortion < og_distortion:
                    og_distortion = new_distortion
                    best_medoid = point  # Set the new medoid to the one with the least distortion
                    swap_occurred = True
            
            medoids[med] = best_medoid
        
        if np.array_equal(old_medoids, medoids):
            break
    
    return assignment, medoids
        

def main():
    # Load the image files directly

    directory = '<directoryname>'  # Get the directory of the script
    K = 32  # Number of clusters
    # Loop through all image files in the directory
    for filename in os.listdir(directory):
        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_file_name = os.path.join(directory, filename)

            im = np.asarray(imageio.imread(image_file_name))

    
            fig, axs = plt.subplots(1, 2)

            # Apply K-medoids
            classes, centers = mykmedoids(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[0].imshow(new_im)
            axs[0].set_title('K-medoids')

            # Apply K-means
            classes, centers = mykmeans(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[1].imshow(new_im)
            axs[1].set_title('K-means')

            plt.show()

if __name__ == '__main__':
    main()

