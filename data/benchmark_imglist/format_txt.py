import os

# To construct the test list
def list_images_in_directory(directory, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    f.write(f"sun/{relative_path} -1\n")

# Example usage
directory = 'YOUR/PATH/TO/data/images_largescale/sun'
output_file = 'imagenet/test_sun.txt'
list_images_in_directory(directory, output_file)