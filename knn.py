from parse import image_parse, label_parse
from sklearn import datasets, neighbors, linear_model, metrics
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import matplotlib.pyplot as plt
import cv2

digits = image_parse("train-images-idx3-ubyte").reshape(60000, 784)
for pic in digits:
    for pixel in pic:
        if pixel > 0:
            pixel = 255
X_train = (digits) / 255.0
y_train = label_parse("train-labels-idx1-ubyte")
X_test = image_parse("t10k-images-idx3-ubyte").reshape(10000, 784)
for pic in X_test:
    for pixel in pic:
        if pixel > 0:
            pixel = 255
X_test = (X_test) / 255.0
y_test = label_parse("t10k-labels-idx1-ubyte")

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
# score = knn.score(X_test, y_test)
# print("Score: " + str(score))

# wrong = 0
# for i in range(0, 10000):
#     prediction = knn.predict(X_test[i].reshape(1, -1))
#     print("Digit Number " + str(i))
#     print("Prediction: " + str(prediction[0]))
#     print("Actual: " + str(y_test[i]))
#     if prediction[0] != y_test[i]:
#         wrong += 1

# print("Wrongs: " + str(wrong))

ax = []
fig = plt.figure(figsize=(20,20))
for num in range(100):
    img = X_test[num].reshape(28, 28)
    ax.append(fig.add_subplot(10, 10, num + 1))
    ax[-1].set_title(str(knn.predict(X_test[num].reshape(1, -1))) + " " + str(y_test[num]))
    ax[-1].axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
fig.tight_layout()
plt.axis('off')
plt.show()

dump(knn, "knn.joblib")

# print("Nearest Neighbor Score: %f" % knn.fit(X_train, y_train).score(X_test, y_test))