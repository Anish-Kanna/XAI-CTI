from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.numpy(), preds)
print(cm)