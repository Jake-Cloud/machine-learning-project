from parse import image_parse, label_parse
from sklearn import datasets, neighbors, linear_model, metrics
from sklearn.neural_network import MLPClassifier
from joblib import dump
import matplotlib.pyplot as plt
import cv2

digits = image_parse("train-images-idx3-ubyte").reshape(60000, 784)
for pic in digits:
    for pixel in pic:
        if pixel > 0:
            pixel = 255

X_train = (255 - digits) / 255.0
y_train = label_parse("train-labels-idx1-ubyte")
X_test = image_parse("t10k-images-idx3-ubyte").reshape(10000, 784)
X_test = (255 - X_test) / 255.0
y_test = label_parse("t10k-labels-idx1-ubyte")

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(1500,), random_state=1)

clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

ax = []
fig = plt.figure(figsize=(20,20))
for num in range(100):
    img = X_test[num].reshape(28, 28)
    ax.append(fig.add_subplot(10, 10, num + 1))
    ax[-1].set_title(str(clf.predict(X_test[num].reshape(1, -1))) + " " + str(y_test[num]))
    ax[-1].axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
fig.tight_layout()
plt.axis('off')
plt.show()

dump(clf, "newclf.joblib")

# print(clf.predict(X_test[0].reshape(1, 784)))
# print(y_test[0])

# expected = y_test
# predicted = clf.predict(X_test)

# print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# img = cv2.imread('4.png')
# img = cv2.resize(img, (28, 28))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# print(img)
# img = img / 255.0
# print(img)
# print("Prediction: " + str(clf.predict(img.reshape(1, 784))))
# plt.imshow(img)
# plt.show()