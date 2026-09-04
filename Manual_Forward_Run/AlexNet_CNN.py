import cv2
import numpy as np

#image Load
image_path = r"C:\Users\Abdullah Hashmi\Downloads\CNN Scratch\Testing images\Face_image1.webp"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

original_image = image.copy()
print(image.shape)

if image is None:
    print("image not found")
    exit()

print("origninal image shape: ", image.shape)

#Convert BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("RGB Image Shape:", image.shape)

#Resize
image = cv2.resize(image,(64,64))
print("Resized RGB image Shape:", image.shape)

#Normalize
image = image.astype(np.float64)/ 255.0
print("Normalized image shape:", image.shape)

#Convolution 1
def convolution(input_data, filters, biases):

    input_height, input_width, input_channels = input_data.shape

    kernel_height, kernel_width, kernel_channels, num_filters = filters.shape

    if input_channels != kernel_channels:
        raise ValueError(
            "Input channels and filter channels do not match"
        )

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    output = np.zeros(
        (output_height, output_width, num_filters),
        dtype=np.float64
    )
    for i in range(output_height):

        # Loop over output width
        for j in range(output_width):

            # Extract a small region from input
            #
            # Example:
            # 3 x 3 x 3
            #
            region = input_data[
                i:i + kernel_height,
                j:j + kernel_width,
                :
            ]

            # Apply every filter
            for f in range(num_filters):

                # Select filter
                kernel = filters[:, :, :, f]
    
              # Element-wise multiplication
                #
                # region:
                # 3 x 3 x channels
                #
                # kernel:
                # 3 x 3 x channels
                #
                # Then sum everything.
                #
                value = np.sum(region * kernel)

                # Add bias
                value += biases[f]

                # Store result
                output[i, j, f] = value

    return output

# RELU
def relu(feature_maps):
    """
    ReLU activation function.

    ReLU(x) = max(0, x)
    """

    return np.maximum(0, feature_maps)

# MAX POOLING
# Pool size:
#
#     2 x 2
#
# Stride:
#
#     2
#
# Important:
#
# Pooling is applied independently to every feature map.
#
# Example:
#
#     62 x 62 x 4
#
# becomes:
#
#     31 x 31 x 4
def max_pool(feature_maps, pool_size=2, stride=2):

    height, width, channels = feature_maps.shape

    output_height = (height - pool_size) // stride + 1
    output_width = (width - pool_size) // stride + 1

    output = np.zeros(
        (output_height, output_width, channels),
        dtype=np.float64
    )

    # Loop through channels
    for c in range(channels):

        # Loop through height
        for i in range(output_height):

            # Loop through width
            for j in range(output_width):

                start_i = i * stride
                start_j = j * stride

                # Extract 2x2 region
                region = feature_maps[
                    start_i:start_i + pool_size,
                    start_j:start_j + pool_size,
                    c
                ]

                # Select maximum value
                output[i, j, c] = np.max(region)

    return output

# SOFTMAX
def softmax(x):
    """
    Convert raw FC outputs into probabilities.
    """
    x = x - np.max(x)

    exp_values = np.exp(x)

    probabilities = exp_values / np.sum(exp_values)

    return probabilities

# WEIGHT INITIALIZATION
# We create random filters.
#
# IMPORTANT:
#
# These are NOT trained weights.
#
# Therefore the final classification result is NOT meaningful.
#
# This program demonstrates the forward-pass architecture.
np.random.seed(42)

# CONV1
# Input:
#
#     64 x 64 x 3
#
# RGB = 3 channels
#
# Conv1:
#
#     4 filters
#     3 x 3
#
# Each filter must have depth 3:
#
#     3 x 3 x 3
#
# Four filters:
#
#     3 x 3 x 3 x 4
#
# Output:
#
#     62 x 62 x 4
conv1_filters = np.random.randn(
    3, 3, 3, 4
) * 0.01

conv1_biases = np.zeros(4)

conv1 = convolution(
    image,
    conv1_filters,
    conv1_biases
)

print("\nAfter Conv1:")
print(conv1.shape)
# RELU 1

relu1 = relu(conv1)

print("After ReLU1:")
print(relu1.shape)

# MAX POOL 1
pool1 = max_pool(
    relu1,
    pool_size=2,
    stride=2
)

print("After Max Pool1:")
print(pool1.shape)

# CONV2
# Input:
#
#     31 x 31 x 4
#
# Because Conv1 created 4 feature maps.
#
# Every Conv2 filter must therefore have:
#
#     3 x 3 x 4
#
# We use 8 filters.
#
# Filter bank:
#
#     3 x 3 x 4 x 8
#
# Output:
#
#     29 x 29 x 8

conv2_filters = np.random.randn(
    3, 3, 4, 8
) * 0.01

conv2_biases = np.zeros(8)

conv2 = convolution(
    pool1,
    conv2_filters,
    conv2_biases
)

print("\nAfter Conv2:")
print(conv2.shape)

# RELU 2
relu2 = relu(conv2)

print("After ReLU2:")
print(relu2.shape)
# MAX POOL 2
pool2 = max_pool(
    relu2,
    pool_size=2,
    stride=2
)

print("After Max Pool2:")
print(pool2.shape)
# CONV3
# Input:
#
#     14 x 14 x 8
#
# Conv3:
#
#     8 filters
#
# Every filter:
#
#     3 x 3 x 8
#
# Output:
#
#     12 x 12 x 8
#
conv3_filters = np.random.randn(
    3, 3, 8, 8
) * 0.01

conv3_biases = np.zeros(8)

conv3 = convolution(
    pool2,
    conv3_filters,
    conv3_biases
)
print("\nAfter Conv3:")
print(conv3.shape)
# RELU 3
relu3 = relu(conv3)

print("After ReLU3:")
print(relu3.shape)
# CONV4
# Input:
#
#     12 x 12 x 8
#
# Filter:
#
#     3 x 3 x 8
#
# Number of filters:
#
#     8
#
# Output:
#
#     10 x 10 x 8
conv4_filters = np.random.randn(
    3, 3, 8, 8
) * 0.01

conv4_biases = np.zeros(8)

conv4 = convolution(
    relu3,
    conv4_filters,
    conv4_biases
)

print("\nAfter Conv4:")
print(conv4.shape)
# RELU 4
relu4 = relu(conv4)

print("After ReLU4:")
print(relu4.shape)

# Input:
#
#     10 x 10 x 8
#
# Filter:
#
#     3 x 3 x 8
#
# Number of filters:
#
#     8
#
# Output:
#
#     8 x 8 x 8

conv5_filters = np.random.randn(
    3, 3, 8, 8
) * 0.01

conv5_biases = np.zeros(8)

conv5 = convolution(
    relu4,
    conv5_filters,
    conv5_biases
)

print("\nAfter Conv5:")
print(conv5.shape)

# RELU 5
relu5 = relu(conv5)

print("After ReLU5:")
print(relu5.shape)

# MAX POOL 3
# Input:
#
#     8 x 8 x 8
#
# Output:
#
#     4 x 4 x 8

pool3 = max_pool(
    relu5,
    pool_size=2,
    stride=2
)

print("\nAfter Max Pool3:")
print(pool3.shape)


# FLATTEN
# Input:
#
#     4 x 4 x 8
#
# Flatten converts this into:
#
#     4 * 4 * 8 = 128
flatten = pool3.flatten()

print("\nAfter Flatten:")
print(flatten.shape)

# FULLY CONNECTED LAYER 1
# ============================================================
#
# 128 inputs
# 64 neurons
fc1_weights = np.random.randn(
    128, 64
) * 0.01

fc1_bias = np.zeros(64)

fc1 = np.dot(
    flatten,
    fc1_weights
) + fc1_bias

# ReLU after FC1
fc1 = relu(fc1)

print("\nAfter FC1:")
print(fc1.shape)

# FULLY CONNECTED LAYER 2
# 64 inputs
# 32 neurons

fc2_weights = np.random.randn(
    64, 32
) * 0.01

fc2_bias = np.zeros(32)

fc2 = np.dot(
    fc1,
    fc2_weights
) + fc2_bias

# ReLU after FC2
fc2 = relu(fc2)

print("\nAfter FC2:")
print(fc2.shape)

# FULLY CONNECTED LAYER 3
# This is the output layer.
#
# Here we use 10 output classes as an example.
#
# Example:
#
# Class 0
# Class 1
# Class 2
# ...
# Class 9
num_classes = 10

fc3_weights = np.random.randn(
    32, num_classes
) * 0.01

fc3_bias = np.zeros(num_classes)

fc3 = np.dot(
    fc2,
    fc3_weights
) + fc3_bias

print("\nAfter FC3:")
print(fc3.shape)

print("Raw FC3 output:")
print(fc3)
# SOFTMAX
# Softmax converts the 10 raw values into probabilities.
#
# Example:
#
# Class 0 = 0.10
# Class 1 = 0.04
# Class 2 = 0.21
# ...
#
# All probabilities add up to approximately 1.
probabilities = softmax(fc3)

print("\nSoftmax probabilities:")

for i, probability in enumerate(probabilities):
    print(
        f"Class {i}: {probability:.6f}"
    )

# FINAL PREDICTION

predicted_class = np.argmax(probabilities)

print("\nFinal Prediction:")
print("Predicted Class:", predicted_class)

print(
    "Confidence:",
    f"{probabilities[predicted_class] * 100:.2f}%"
)


# VERIFY SOFTMAX
print(
    "\nSum of Softmax probabilities:",
    np.sum(probabilities)
)

# DISPLAY INPUT IMAGE
# OpenCV expects BGR when displaying.
#
# Therefore convert RGB back to BGR.
display_image = cv2.cvtColor(
    (image * 255).astype(np.uint8),
    cv2.COLOR_RGB2BGR
)
display_large = cv2.resize(
    display_image,
    (400, 400),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow("Mini AlexNet Input Image", display_large)
cv2.waitKey(0)
cv2.destroyAllWindows()