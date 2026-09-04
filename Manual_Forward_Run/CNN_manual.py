import cv2
import numpy as np

#image Load
image = cv2.imread(r"C:\Users\Abdullah Hashmi\Downloads\CNN_From_Scratch\Testing images\Face_image1.webp", cv2.IMREAD_GRAYSCALE)
print(image.shape)

if image is None:
    print("image not found")
    exit()

print("origninal image shape: ", image.shape)

#Convolution
def convolution(image):
    kernel = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], dtype = np.float64)

    rows,cols = image.shape
    output = np.zeros((rows-2, cols-2))

    for i in range(rows-2):
        for j in range(cols-2):
            region = image[i:i+3, j:j+3]
            value = np.sum(region * kernel)
            output[i,j] = value

    return output
feature_map = convolution(image)
print("\n After Convolution:")
print(feature_map.shape)

#RELU
def relu(feature_map):
    return np.maximum(0,feature_map)

relu_output = relu(feature_map)

print("\n After Relu")
print(relu_output.shape)

#MAx Pooling
def max_pool(feature_map):
    rows, cols = feature_map.shape
    output = np.zeros((rows//2, cols//2))
    for i in range(0, rows-1, 2):
        for j in range(0, cols-1, 2):
            region = feature_map[i:i+2, j:j+2]
            output[i//2,j//2] = np.max(region)
    return output

pooled = max_pool(relu_output)

print("\nAfter Pooling:")
print(pooled.shape)

#Flatten
flatten = pooled.flatten()

print("\n After Flatten:")
print(flatten.shape)

#Fully Connecnted
weights = np.random.rand(flatten.shape[0],1)
bias = np.random.rand(1)
fc_output = np.dot(flatten,weights) + bias
print("\nAfter Fully Connected:")
print(fc_output)

#Sigmoid Output
def sigmoid(x):
     return 1 / (1 + np.exp(-x))
prediction = sigmoid(fc_output)
print("\nFinal Prediction:")
print(prediction)

#Display Image
cv2.imshow("Input Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()