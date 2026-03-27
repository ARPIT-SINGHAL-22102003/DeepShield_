from mesonet import predict_image, test_predict_repeat
import os

# Test on some images if available
test_images = []
if os.path.exists('dataset_raw/real_vs_fake/real-vs-fake/train/real'):
    real_files = [f for f in os.listdir('dataset_raw/real_vs_fake/real-vs-fake/train/real') if f.endswith('.jpg')][:2]
    test_images.extend([os.path.join('dataset_raw/real_vs_fake/real-vs-fake/train/real', f) for f in real_files])

if os.path.exists('dataset_raw/real_vs_fake/real-vs-fake/train/fake'):
    fake_files = [f for f in os.listdir('dataset_raw/real_vs_fake/real-vs-fake/train/fake') if f.endswith('.jpg')][:2]
    test_images.extend([os.path.join('dataset_raw/real_vs_fake/real-vs-fake/train/fake', f) for f in fake_files])

if test_images:
    for img in test_images:
        test_predict_repeat(img, 5)
else:
    print("No test images found in dataset_raw/real_vs_fake/real-vs-fake/train/")