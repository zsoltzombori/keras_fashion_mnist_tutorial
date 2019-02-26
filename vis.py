import matplotlib
matplotlib.use('Agg') # if running python from a non graphical shell, uncommment this line
import matplotlib.pyplot as plt
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

def vis_learning_curves(histories, fileName):
    acc_list=[]
    val_acc_list=[]
    loss_list=[]
    val_loss_list=[]
    for history in histories:
        acc_list.append(history.history['acc'])
        val_acc_list.append(history.history['val_acc'])
        loss_list.append(history.history['loss'])
        val_loss_list.append(history.history['val_loss'])
    acc_list = np.concatenate(acc_list)
    val_acc_list = np.concatenate(val_acc_list)
    loss_list = np.concatenate(loss_list)
    val_loss_list = np.concatenate(val_loss_list)
    NB_EPOCHS = len(acc_list)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(1, NB_EPOCHS+1), loss_list, 'r--', label="train")
    plt.plot(range(1, NB_EPOCHS+1), val_loss_list, 'b--', label="validation")
    plt.ylabel("loss")
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    plt.subplot(212)
    plt.plot(range(1, NB_EPOCHS+1), acc_list, 'r--', label="train")
    plt.plot(range(1, NB_EPOCHS+1), val_acc_list, 'b--', label="validation")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')

    plt.savefig(fileName)
