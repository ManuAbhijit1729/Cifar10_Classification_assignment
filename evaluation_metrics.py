
import numpy as np
from sklearn.metrics import confusion_matrix

# Macro Definition
# To create confusion matrix using sklearn
def create_confusion_matrix(label, output, num_classes):
    return confusion_matrix(
            label.detach().cpu().numpy().astype(np.int8).reshape(-1),
            output.detach().cpu().numpy().astype(np.int8).reshape(-1),
            labels=range(num_classes)
        )
#Function to compute classwise evaluation scores(P/R/F1/Accuracy) and then take arithmetic mean to compute overall macro scores.
#Same as using sklearn.metrics.precision_score/recall_score etc. and using average='macro' in the said function.
def calculate(confusion_matrix,NO_OF_CLASSES=2, return_metrics = False):

    total_predictions = np.sum(confusion_matrix)
    mean_accuracy = mean_F1 = mean_precision = mean_recall = 0

    F1_scores = []

    for class_id in range(0, NO_OF_CLASSES):
        tp = confusion_matrix[class_id, class_id]
        fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1 :, class_id])
        fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1 :])
        tn = total_predictions - tp - fp - fn
        
        delta = 1e6 # constant division factor to ensure values are in the precision range
        epsilon = 1e-6 # adding epsilon to the denominators avoid divive by zero error
        
        # print(f"class_id:{class_id}, tp:{tp}, fp:{fp} ,fn:{fn}, tn:{tn}") #to print classwise scores
        
        accuracy = ((tp/delta) + (tn/delta)) / ((tn/delta) + (fn/delta) + (tp/delta) + (fp/delta))
        mean_accuracy += accuracy
        
        recall = (tp/delta)/((tp/delta) + (fn/delta) + epsilon)  # adding epsilon to avoid divive by zero error, dividing each entity in num/den by delta to ensure values are in precision range. 
        mean_recall += recall
        
        precision = (tp/delta)/((tp/delta)+(fp/delta) + epsilon) # adding epsilon to avoid divive by zero error
        mean_precision += precision
    

        if (((tp/delta) + (fp/delta) + (fn/delta)) != 0):
            F1_Score = ((2 * tp)/delta) / ((2 * tp)/delta + fp/delta + fn/delta)
        else:
            # When there are no positive samples and model is not having any false positive, we can not judge F1_Score score
            # In this senario we assume worst case F1_Score score. This also avoids 0/0 condition .
            F1_Score = 0.0
    
        mean_F1 += F1_Score
        F1_scores.append(F1_Score)

    mean_accuracy = mean_accuracy / (NO_OF_CLASSES) #Macro scores (Mean across classes) Same as sklearn using average='macro'
    mean_F1 = mean_F1 / (NO_OF_CLASSES)
    mean_precision = mean_precision / (NO_OF_CLASSES)
    mean_recall = mean_recall / (NO_OF_CLASSES)

    if return_metrics:
        return F1_scores, mean_precision, mean_recall
    else:
        return mean_precision, mean_recall, mean_accuracy, mean_F1
