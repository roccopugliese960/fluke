
import torch
import torchvision
from torchvision import transforms
import os
import sys  # Import the 'sys' module to read command-line arguments
from fluke.nets import MNIST_2NN

# --- 1. CONFIGURATION IS NOW DYNAMIC ---
# The MODEL_PATH is no longer hardcoded. It will be read from the command line.

TRIGGER_LABEL = 7
TARGET_LABEL = 1

def verify_backdoor(model_path: str):
    """
    Loads a trained global model from a specific path and tests for the backdoor.
    
    Args:
        model_path (str): The full path to the saved server model file (e.g., r0050_server.pth).
    """
    print("--- Backdoor Verification Script ---")

    # --- 2. LOAD THE MODEL FROM THE PROVIDED PATH ---
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        print("Please provide a valid path as a command-line argument.")
        return

    print(f"Loading model from: {model_path}")
    device = torch.device("cpu")
    model = MNIST_2NN()
    
    try:
        server_checkpoint = torch.load(model_path)
        model_state_dict = server_checkpoint['model']
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"ERROR: Failed to load the model. Ensure the file is a valid Fluke server checkpoint.")
        print(f"   Details: {e}")
        return

    model.to(device)
    model.eval()

    # --- 3. PREPARE THE DATA ---
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    test_image, original_label = None, -1
    for img, lbl in test_dataset:
        if lbl == TRIGGER_LABEL:
            test_image = img
            original_label = lbl
            break
    
    if test_image is None:
        print(f"Error: Could not find an image with label {TRIGGER_LABEL} in the test set.")
        return

    test_image = test_image.unsqueeze(0).to(device)

    # --- 4. TEST ON CLEAN IMAGE ---
    with torch.no_grad():
        clean_output = model(test_image)
        clean_prediction = torch.argmax(clean_output, dim=1).item()
    
    print(f"\n1. Testing on a clean, original image of a '{original_label}'...")
    print(f"   Prediction on CLEAN image: {clean_prediction}")
    if clean_prediction == original_label:
        print("SUCCESS: The model correctly classifies the clean image.")
    else:
        print("WARNING: The model misclassifies the clean image.")

    # --- 5. TEST ON TRIGGERED IMAGE ---
    triggered_image = test_image.clone()
    triggered_image[0, 0:4, 0:4] = 1.0 
    
    with torch.no_grad():
        triggered_output = model(triggered_image)
        triggered_prediction = torch.argmax(triggered_output, dim=1).item()

    print(f"\n2. Applying backdoor trigger to the image...")
    print(f"   Prediction on TRIGGERED image: {triggered_prediction}")
    if triggered_prediction == TARGET_LABEL:
        print(f"ATTACK PRESENT: The model is backdoored.")
    else:
        print(f"ATTACK NEUTRALIZED: The model correctly ignored the trigger. The defense worked!")

if __name__ == "__main__":
    # The `sys.argv` list contains command-line arguments.
    # sys.argv[0] is the script name itself (e.g., "verify_backdoor.py").
    # sys.argv[1] is the first argument we provide.
    if len(sys.argv) < 2:
        print("ERROR: Missing argument.")
        print("Usage: python verify_backdoor.py <path_to_your_model.pth>")
    else:
        model_to_test = sys.argv[1]
        verify_backdoor(model_to_test)