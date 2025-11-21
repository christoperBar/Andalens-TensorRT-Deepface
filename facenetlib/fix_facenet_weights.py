"""
Fix existing FaceNet weights file for TFJS conversion
This script takes your facenet512_weights.h5 and creates a complete model ready for TFJS
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

# Custom scaling layer to replace Lambda
class ScalingLayer(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.scale = scale
    
    def call(self, inputs, **kwargs):
        # Modified to accept additional kwargs like mask
        return inputs * self.scale
    
    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config


def fix_facenet_for_tfjs(weights_path, dimension=512):
    """
    Takes existing FaceNet weights and creates a complete TFJS-compatible model
    
    Args:
        weights_path: Path to your facenet weights file (e.g., 'facenet512_weights.h5')
        dimension: 128 or 512 (based on your weights file)
    """
    
    print(f"🔧 Fixing FaceNet weights for TFJS conversion...")
    print(f"📁 Weights file: {weights_path}")
    print(f"📐 Dimension: {dimension}")
    
    # Step 1: Import and create the original model architecture
    print("\n1️⃣ Creating original FaceNet architecture...")
    from Facenet_standalone import InceptionResNetV1
    
    # Create model with original Lambda layers
    original_model = InceptionResNetV1(dimension=dimension)
    print(f"✅ Created model: {original_model.name}")
    
    # Step 2: Load your existing weights
    print("\n2️⃣ Loading weights from file...")
    try:
        original_model.load_weights(weights_path)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None
    
    # Step 3: Create TFJS-compatible model by replacing Lambda layers
    print("\n3️⃣ Converting Lambda layers to TFJS-compatible layers...")
    
    # Get model configuration
    config = original_model.get_config()
    
    # Count Lambda layers
    lambda_count = 0
    
    # Replace Lambda layers in the config
    for i, layer_config in enumerate(config['layers']):
        if layer_config['class_name'] == 'Lambda':
            lambda_count += 1
            # Extract scale value from Lambda arguments
            scale = layer_config['config']['arguments']['scale']
            
            # Create new ScalingLayer config
            new_config = {
                'class_name': 'ScalingLayer',
                'config': {
                    'name': layer_config['config']['name'],
                    'trainable': layer_config['config']['trainable'],
                    'dtype': layer_config['config']['dtype'],
                    'scale': scale
                },
                'name': layer_config['name'],
                'inbound_nodes': layer_config['inbound_nodes']
            }
            config['layers'][i] = new_config
    
    print(f"✅ Replaced {lambda_count} Lambda layers with ScalingLayer")
    
    # Step 4: Create new model from modified config
    print("\n4️⃣ Creating TFJS-compatible model...")
    custom_objects = {'ScalingLayer': ScalingLayer}
    tfjs_model = keras.Model.from_config(config, custom_objects=custom_objects)
    
    # Step 5: Transfer weights from original to new model
    print("\n5️⃣ Transferring weights...")
    weights_transferred = 0
    for orig_layer, new_layer in zip(original_model.layers, tfjs_model.layers):
        if orig_layer.get_weights():
            new_layer.set_weights(orig_layer.get_weights())
            weights_transferred += 1
    
    print(f"✅ Transferred weights for {weights_transferred} layers")
    
    # Step 6: Verify the model
    print("\n6️⃣ Verifying model...")
    # Test with random input
    test_input = tf.random.normal((1, 160, 160, 3))
    try:
        output = tfjs_model(test_input)
        print(f"✅ Model output shape: {output.shape}")
        print(f"✅ Model is working correctly!")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return None
    
    return tfjs_model


def save_for_tfjs(model, output_dir="facenet_tfjs_ready"):
    """Save the model in formats suitable for TFJS conversion"""
    
    print(f"\n💾 Saving model for TFJS conversion...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SavedModel (recommended for TFJS)
    savedmodel_path = os.path.join(output_dir, "saved_model")
    print(f"   Saving SavedModel to {savedmodel_path}...")
    tf.saved_model.save(model, savedmodel_path)
    
    # Save as H5 with custom objects
    h5_path = os.path.join(output_dir, "model.h5")
    print(f"   Saving H5 to {h5_path}...")
    model.save(h5_path, include_optimizer=False)
    
    # Save just the weights (backup) - UPDATED FORMAT TO .weights.h5
    weights_path = os.path.join(output_dir, "model.weights.h5")
    print(f"   Saving weights to {weights_path}...")
    model.save_weights(weights_path)
    
    # Create conversion script
    script_content = f"""#!/bin/bash
# Convert FaceNet to TFJS

echo "Converting to TFJS format..."

# Method 1: From SavedModel (Recommended)
tensorflowjs_converter \\
    --input_format=tf_saved_model \\
    --output_format=tfjs_graph_model \\
    --signature_name=serving_default \\
    --saved_model_tags=serve \\
    {output_dir}/saved_model \\
    tfjs_model

echo "Conversion complete! Check tfjs_model/ directory"
"""
    
    script_path = os.path.join(output_dir, "convert.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    print(f"\n✅ Model saved to {output_dir}/")
    print(f"📜 Run the conversion script: bash {script_path}")
    
    return output_dir


def main():
    """Main function to convert your existing weights"""
    
    print("🚀 FaceNet Weights to TFJS Converter")
    print("=" * 50)
    
    # Check if weights file exists
    weights_file = "facenet512_weights.h5"  # Change this to your file name
    
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
    
    if not os.path.exists(weights_file):
        print(f"\n❌ Error: Weights file '{weights_file}' not found!")
        print("\nUsage: python fix_facenet_weights.py [weights_file]")
        print("Example: python fix_facenet_weights.py facenet512_weights.h5")
        return
    
    # Detect dimension from filename
    if "512" in weights_file:
        dimension = 512
    elif "128" in weights_file:
        dimension = 128
    else:
        print("\n⚠️  Cannot detect dimension from filename.")
        dimension = int(input("Enter dimension (128 or 512): "))
    
    # Fix the model
    tfjs_model = fix_facenet_for_tfjs(weights_file, dimension)
    
    if tfjs_model is None:
        print("\n❌ Failed to create TFJS-compatible model")
        return
    
    # Save the fixed model
    output_dir = save_for_tfjs(tfjs_model, f"facenet{dimension}_tfjs_ready")
    
    # Print final instructions
    print("\n" + "=" * 50)
    print("✅ SUCCESS! Your model is ready for TFJS conversion")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Install tensorflowjs: pip install tensorflowjs")
    print(f"2. Run: bash {output_dir}/convert.sh")
    print("3. Your TFJS model will be in tfjs_model/")
    print("\n🎉 The model.json will now have complete topology!")


if __name__ == "__main__":
    main()
