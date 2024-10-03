Step 1: Install the necessary libraries
You need to install the following libraries:

```pip install torch onnx onnx-tf tensorflow```

Step 2: Save the PyTorch Model to ONNX Format

Execute ```pytorch_to_onnx.py```
This will export both model_vol and model_sal to ONNX format. (You can change this file to export the partial model.)

Step 3: Convert ONNX to TensorFlow

```onnx-tf convert -i model_vol.onnx -o model_vol_tf```

```onnx-tf convert -i model_sal.onnx -o model_sal_tf```

Step 4: Load the TensorFlow Model
```
import tensorflow as tf

model_vol_tf = tf.saved_model.load("model_vol_tf")
model_sal_tf = tf.saved_model.load("model_sal_tf")

# Example usage of the TensorFlow models
dummy_input = tf.random.normal([1, 255, 255, 3]) 

output_vol = model_vol_tf(dummy_input)
output_sal = model_sal_tf(dummy_input)

```
