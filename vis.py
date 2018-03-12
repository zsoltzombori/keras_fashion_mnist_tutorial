import numpy as np
import scipy.misc
from collections import defaultdict

# Visualizing the classification:
def vis_classification(X, y_true, y_pred, bucket_size=10, nb_classes=10, image_size=28, file_name="vis.png"):
    vis_image_size = nb_classes * image_size * bucket_size
    vis_image = 255 * np.ones((vis_image_size, vis_image_size), dtype='uint8')
    example_counts = defaultdict(int)
    for (predicted_tag, actual_tag, image) in zip(y_pred, y_true, X):
        image = ((1 - image) * 255).reshape((image_size, image_size)).astype('uint8')
        example_count = example_counts[(predicted_tag, actual_tag)]
        if example_count >= bucket_size**2:
            continue
        tilepos_x = bucket_size * predicted_tag
        tilepos_y = bucket_size * actual_tag
        tilepos_x += example_count % bucket_size
        tilepos_y += example_count // bucket_size
        pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
        vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size] = image
        example_counts[(predicted_tag, actual_tag)] += 1

    vis_image[::image_size * bucket_size, :] = 0
    vis_image[:, ::image_size * bucket_size] = 0
    scipy.misc.imsave(file_name, vis_image)
