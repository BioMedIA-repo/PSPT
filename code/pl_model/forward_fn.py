import torch


model_to_classifier_type = {
    "clam_sb": "clam", "clam_mb": "clam",
}


def get_classifer_fuc(classifier_type):
    if classifier_type == 'clam':
        return clam_forward
    else:
        raise NotImplementedError


def clam_forward(data, classifier, loss, num_classes, label=None):
    logits, Y_prob, Y_hat, _, instance_dict = classifier(data, label=label, instance_eval=True)
    loss = loss(logits, label)
    instance_loss = instance_dict['instance_loss']
    total_loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_loss
    return logits, total_loss, Y_prob
