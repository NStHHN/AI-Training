import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import itertools
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims
from tensorflow.keras.models import Model
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tf_explain.core.grad_cam import GradCAM



def plot_images(path):
    plt.figure(figsize=(20,20))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(cv2.imread(path[i]))
    plt.show()
def get_confusion_matrix(data_path, N, gen, batch_size, model):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=(224, 224), shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm   
def get_confusion_matrix_gray(data_path, N, gen, batch_size, model):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=(224, 224), shuffle=False, batch_size=batch_size * 2,color_mode='grayscale'):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm   
def get_confusion_matrix_bin(data_path, N, gen, batch_size, model):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    ind = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=(224, 224), shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)

        p = np.around(p).astype(int).transpose()[0]
        p = to_categorical(p)

        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
                    
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break


    cm = confusion_matrix(targets, predictions)
    return cm

def plot_confusion_matrix(c, valid_c, train):
    labels = [None] * len(train.class_indices)
    for k, v in train.class_indices.items():
        labels[v] = k
    plt.figure(figsize=(10,5))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(c, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Train confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = 'd'
    thresh = c.max() / 2.
    for i, j in itertools.product(range(c.shape[0]), range(c.shape[1])):
        plt.text(j, i, format(c[i, j], fmt),
               horizontalalignment="center",
               color="white" if c[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

    # ----------------
    ax = plt.subplot(1, 2, 2)
    plt.imshow(valid_c, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = 'd'
    thresh = c.max() / 2.
    for i, j in itertools.product(range(valid_c.shape[0]), range(c.shape[1])):
        plt.text(j, i, format(valid_c[i, j], fmt),
               horizontalalignment="center",
               color="white" if c[i, j] > thresh else "black")

    plt.tight_layout(pad=5.0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    

def show_feature_maps_gray(model,layer_num, img, first_layer=False):
    # redefine model to output right after the first hidden layer
    model_part = Model(inputs=model.inputs, outputs=model.layers[layer_num].output)
    model_part.summary()
    # load the image with the required shape
    img = load_img(img, target_size=(224, 224), color_mode='grayscale')
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    #img = preprocess_input(img)
    img = img*1./255
    # get feature map for first hidden layer
    feature_maps = model_part.predict(img)
    # plot all 64 maps in an 8x8 squares
    if first_layer == True:
        rows = 4
        plt.figure(figsize=(20,10))
    else:
        plt.figure(figsize=(20,20))
        rows = 8
    columns = 8
    ix = 1
    image_added = np.zeros(feature_maps[0, :, :, 26].shape)
    
    for _ in range(rows):
        for _ in range(columns):
            # specify subplot and turn of axis

            ax = plt.subplot(rows, columns, ix) 
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            #print(feature_maps[0, :, :, 26-1].shape,np.min(feature_maps[0, :, :, 26-1]),np.max(feature_maps[0, :, :, 26-1]))
            image_added += feature_maps[0, :, :, ix-1]
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    image_added /= np.max(image_added)
    plt.show()
    return feature_maps, image_added
def show_feature_maps(model,layer_num, img, first_layer=False):
    # redefine model to output right after the first hidden layer
    model_part = Model(inputs=model.inputs, outputs=model.layers[layer_num].output)
    model_part.summary()
    # load the image with the required shape
    img = load_img(img, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    #img = preprocess_input(img)
    img = img*1./255
    # get feature map for first hidden layer
    feature_maps = model_part.predict(img)
    # plot all 64 maps in an 8x8 squares
    if first_layer == True:
        rows = 4
        plt.figure(figsize=(20,10))
    else:
        plt.figure(figsize=(20,20))
        rows = 8
    columns = 8
    ix = 1
    image_added = np.zeros(feature_maps[0, :, :, 26].shape)
    
    for _ in range(rows):
        for _ in range(columns):
            # specify subplot and turn of axis

            ax = plt.subplot(rows, columns, ix) 
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            #print(feature_maps[0, :, :, 26-1].shape,np.min(feature_maps[0, :, :, 26-1]),np.max(feature_maps[0, :, :, 26-1]))
            image_added += feature_maps[0, :, :, ix-1]
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    image_added /= np.max(image_added)
    plt.show()
    return feature_maps, image_added

def plot_heatmap(img, heatMap):
    plt.figure(figsize=(20,20))
    img_for_heatmap = cv2.imread(img)
    heatmap = cv2.resize(heatMap, (img_for_heatmap.shape[1], img_for_heatmap.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = (heatmap * hif + img_for_heatmap).squeeze()
    
    output = 'output.jpeg'
    cv2.imwrite(output, superimposed_img)

    img_for_heatmap=mpimg.imread(output)
    ax = plt.subplot(1, 2,1)
    plt.imshow(heatMap)
    ax = plt.subplot(1, 2,2)
    plt.imshow(img_for_heatmap)
    
    plt.axis('off')
    plt.show()
    

def predict_and_plot_gray(model,images):
    plt.figure(figsize=(20,5))
    for ix in range(5):
        
        img = image.load_img(images[ix],target_size=(224,224),color_mode='grayscale')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        x = x*1./255
        pred = model.predict(x)

        pred_id = np.argmax(pred, axis=1)
        pred = np.max(pred,axis=1)*100

        if pred_id == [0]:
            label = "n.i.O" 
            confidence = str(pred[0]) +"%"
            fontcolor = "red"
            
        if pred_id == [1]:
            label = "i.O"
            fontcolor = "green"
        confidence = str(pred[0]) +"%"
        x = np.squeeze(x, axis=0)
        x = np.squeeze(x)
        ax = plt.subplot(1, 5, ix+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.text(10, x.shape[0]-10, label, fontsize=30,color=fontcolor)
        plt.text(10, x.shape[0]+40, confidence, fontsize=15,color="black")
        plt.imshow(x,'gray')
    plt.show()
def predict_and_plot(model,images):
    plt.figure(figsize=(20,5))
    for ix in range(5):
        
        img = image.load_img(images[ix],target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        x = x*1./255
        pred = model.predict(x)

        pred_id = np.argmax(pred, axis=1)
        pred = np.max(pred,axis=1)*100

        if pred_id == [0]:
            label = "n.i.O" 
            confidence = str(pred[0]) +"%"
            fontcolor = "red"
            
        if pred_id == [1]:
            label = "i.O"
            fontcolor = "green"
        confidence = str(pred[0]) +"%"
        x = np.squeeze(x, axis=0)

        ax = plt.subplot(1, 5, ix+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.text(10, x.shape[0]-10, label, fontsize=30,color=fontcolor)
        plt.text(10, x.shape[0]+40, confidence, fontsize=15,color="black")
        plt.imshow(x)
    plt.show()
def plot_grad_cam_gray(model,images):

    plt.figure(figsize=(20,20))
    explainer = GradCAM()
    ix = 1
    square = 4
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            #print(feature_maps[0, :, :, 26-1].shape,np.min(feature_maps[0, :, :, 26-1]),np.max(feature_maps[0, :, :, 26-1]))
            img = load_img(images[ix], target_size=(224, 224),color_mode='grayscale')
            # convert the image to an array
            img = img_to_array(img)
            # expand dimensions so that it represents a single 'sample'
            #img = expand_dims(img, axis=0)
            img = img*1./255
            data = ([img], None)
            grid = explainer.explain(data, model, class_index=0) 
            plt.imshow(grid)
            ix += 1
def plot_grad_cam(model,images):

    plt.figure(figsize=(20,20))
    explainer = GradCAM()
    ix = 1
    square = 4
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            #print(feature_maps[0, :, :, 26-1].shape,np.min(feature_maps[0, :, :, 26-1]),np.max(feature_maps[0, :, :, 26-1]))
            img = load_img(images[ix], target_size=(224, 224))
            # convert the image to an array
            img = img_to_array(img)
            # expand dimensions so that it represents a single 'sample'
            #img = expand_dims(img, axis=0)
            img = img*1./255
            data = ([img], None)
            grid = explainer.explain(data, model, class_index=0) 
            plt.imshow(grid)
            ix += 1