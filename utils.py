import matplotlib.pyplot as plt
import codecs
import itertools
from sklearn.metrics import confusion_matrix
from IPython.display import clear_output
import numpy as np
import tensorflow as tf
from IPython.display import display, HTML
import os
tf.get_logger().setLevel('ERROR')
def rescale_score_by_abs (score, max_score, min_score):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)    
  
      
def getRGB (c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

     
def span_word (word, score, colormap):
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"
def html_heatmap (words, scores, cmap_name="bwr"):
    """
    Return word-level heatmap in HTML format,
    with words being the list of words (as string),
    scores the corresponding list of word-level relevance values,
    and cmap_name the name of the matplotlib diverging colormap.
    """
    
    colormap  = plt.get_cmap(cmap_name)
     
    assert len(words)==len(scores)
    max_s     = max(scores)
    min_s     = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score       = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + span_word(w, score, colormap) + " "
    
    return output_text + "\n"
def get_test_sentence(sent_idx):
    """Returns an SST test set sentence and its true label, sent_idx must be an integer in [1, 2210]"""
    idx = 1
    with codecs.open("sequence_test.txt", 'r', encoding='utf8') as f:
        for line in f:
            line          = line.rstrip('\n')
            line          = line.split('\t')
            true_class    = int(line[0])-1         # true class
            words         = line[1].split(' | ')   # sentence as list of words
            if idx == sent_idx:
                return words, true_class
            idx +=1
            


def plot_confusion_matrix(c,label_map):
    plt.figure(figsize=(10,5))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(c, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_map))
    plt.xticks(tick_marks, label_map, rotation=45)
    plt.yticks(tick_marks, label_map)

    fmt = 'd'
    thresh = c.max() / 2.
    for i, j in itertools.product(range(c.shape[0]), range(c.shape[1])):
        plt.text(j, i, format(c[i, j], fmt),
               horizontalalignment="center",
               color="white" if c[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()


def get_confusion_matrix(w2v,model):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix")
    predictions = []
    targets = []
    for i in range(2210):

        if i % 50 == 0:
            clear_output(wait=True)
            print(i)
        tokens, label =  get_test_sentence(i+1)
        vecs = np.array([w2v[t] for t in tokens])
        vecs = tf.convert_to_tensor(vecs, dtype=tf.float32)
        # predict label
        y = model.predict(vecs[np.newaxis,:])
        predictions.append(np.argmax(y))
        targets.append(label)


            
            
    cm = confusion_matrix(targets, predictions)
    return cm   
def get_and_process_sentence(index,w2v,lrp_model, label_map):
    tokens, label =  get_test_sentence(index)
    vecs = np.array([w2v[t] for t in tokens])
    # predict label
    y_lrpnet, _, _ = lrp_model.full_pass(vecs[np.newaxis,:])
    # explain the classification
    eps = 1e-3
    bias_factor = 0.0
    # by setting y=None, the relevances will be calculated for the predicted class of the sample. We recommend this
    # usage, however, if you are interested in the relevances towards the 1st class, you could use y = np.array([1])
    explanation, Rest= lrp_model.lrp(vecs[np.newaxis,:], eps=eps, bias_factor=bias_factor)
    word_relevances = tf.reduce_sum(explanation, axis=2)
    display(HTML(html_heatmap(tokens, word_relevances[0])))
    print("\nReal Label: {} ({})".format( label, label_map[label]))
    print("Predicted Label: {} ({})\n".format(np.argmax(y_lrpnet), label_map[np.argmax(y_lrpnet)]))