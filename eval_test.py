from sklearn.metrics import precision_recall_fscore_support

with open('token_eval.txt', 'r') as token_test, open('label_eval.txt', 'r') as label_test:
    labels_actual = [item.strip().split(' ')[1] for item in token_test]
    labels_predict = [item.strip() for item in label_test]
    result = precision_recall_fscore_support(labels_actual, labels_predict,
                                             labels=["B-PER", "M-PER", "E-PER", "B-ROLE", "M-ROLE", "E-ROLE"],
                                             average='macro')
    print(result)
