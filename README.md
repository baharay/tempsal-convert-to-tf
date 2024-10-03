# Step 1: Install the necessary libraries
You need to install the following libraries:

```pip install torch onnx onnx-tf tensorflow```

# Step 2: Save the PyTorch Model to ONNX Format

Execute ```pytorch_to_onnx.py```
This will export both model_vol and model_sal to ONNX format. (You can change this file to export the partial model.)

# Step 3: Convert ONNX to TensorFlow

```onnx-tf convert -i model_vol.onnx -o model_vol_tf```

```onnx-tf convert -i model_sal.onnx -o model_sal_tf```

# Step 4: Load the TensorFlow Model
```
import tensorflow as tf

model_vol_tf = tf.saved_model.load("model_vol_tf")
model_sal_tf = tf.saved_model.load("model_sal_tf")

# Example usage of the TensorFlow models
dummy_input = tf.random.normal([1, 255, 255, 3]) 

output_vol = model_vol_tf(dummy_input)
output_sal = model_sal_tf(dummy_input)
```


# Shape of the inputs and outputs

Shape after deconv_layer0 (out5): torch.Size([1, 512, 16, 16])

Shape after concatenating out5 and out4: torch.Size([1, 2672, 16, 16])

Shape after deconv_layer1: torch.Size([1, 256, 32, 32])

Shape after concatenating x and out3: torch.Size([1, 1336, 32, 32])

Shape after deconv_layer2: torch.Size([1, 270, 64, 64])

Shape after concatenating x and out2: torch.Size([1, 540, 64, 64])

Shape after deconv_layer3: torch.Size([1, 96, 128, 128])

Shape after concatenating x and out1: torch.Size([1, 192, 128, 128])

Shape after deconv_layer4: torch.Size([1, 128, 256, 256])

Shape after deconv_layer5: torch.Size([1, 1, 256, 256])

Final shape after squeeze: torch.Size([1, 256, 256])
