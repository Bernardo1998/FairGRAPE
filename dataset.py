import PIL
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import copy
from PIL import Image
from pathlib import Path
from collections import Counter
import imgaug as ia
from imgaug import augmenters as iaa
import dlib
import time
import os
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Split into train/validation/test sets so that all faces from one image belong to the same set
def split_image_name(val):
    return val.split('/')[-1]

def relabel(frame, seven_races=True, drop_race=False):

	# relabel races for training
	if 'race' in frame.columns:
		frame.loc[frame['race'] == 'White', 'race'] = 0
		frame.loc[frame['race'] == 'Black', 'race'] = 1
		frame.loc[frame['race'] == 'Latino_Hispanic', 'race'] = 2
		frame.loc[frame['race'] == 'East Asian', 'race'] = 3
		frame.loc[frame['race'] == 'Southeast Asian', 'race'] = 4
		frame.loc[frame['race'] == 'Indian', 'race'] = 5
		frame.loc[frame['race'] == 'Middle Eastern', 'race'] = 6

	# gender label
	if 'gender' in frame.columns:
		frame.loc[frame['gender'] == 'Male', 'gender'] = 0
		frame.loc[frame['gender'] == 'Female', 'gender'] = 1

	## ONLY RUN THIS CELL TO TRAIN A MODEL WITH LESS THAN 7 RACES
	if not seven_races and 'race' in frame.columns:
		# Here, we compare our dataset with LFWA+ (no Indian, Latino) & UTK (no Latino)
		#For all datsets:
		# White 0, Black 1, Asian 2, Indian 3
		# drop latino (2) and middle-east (6)
		frame.loc[frame['race'].isin([2, 6]), 'race'] = -1
		# merge east asian (3) with south east asian (4) 
		frame.loc[frame['race'].isin([3,4]), 'race'] = 2
		# relabel Indian (5) to all dataset label of Indian (3)
		frame.loc[frame['race'].isin([5]), 'race'] = 3
		# Drop latino
		frame = frame[frame['race']!=-1]
		# Drop Indian (for comparison with LFWA+ only)
		#frame = frame[frame['race']!=3]
		Counter(frame.race)

	if drop_white and 'race' in frame.columns:
		frame = frame[frame['race']!=0].reset_index(drop=True)
		#frame.loc[frame['race'] != 0, 'race'] = frame.loc[frame['race'] != 0, 'race'] - 1

	# Different behaviors of drop_race:
	# =0: nothing
	# =1/2/3/4: drop white/black/asian/indian
	# =11/12/13/14: keep only white/black/asian/indian
	if drop_race and 'race' in frame.columns:
		if drop_race in [1,2,3,4]:
			drop_race -= 1
			frame = frame[frame['race']!=drop_race].reset_index(drop=True)
		elif drop_race in [11,12,13,14]:
			drop_race -= 11
			frame = frame[frame['race']==drop_race].reset_index(drop=True)

	return frame

def make_frame(csv, new_face_dir, train_pct = 0.8, seven_races=True,drop_race=False):
	device = torch.device('cuda:0')
	frame = pd.read_csv(csv)
	frame.head()

	frame = relabel(frame, seven_races,drop_white)

	# Change face_name_align if the images are now stored in a different dir
	# Also make sure all faces are found and can be
	if new_face_dir:
		initial_rows = frame.shape[0]
		faces = set(os.listdir(new_face_dir))
		faces_found = 0
		new_face_name = []
		face_found_mask = []
		for i in range(frame.shape[0]):
			face_name_align = split_image_name(frame['face_name_align'][i])
			face_found_mask.append(face_name_align in faces)
			if face_name_align in faces:
				new_path = os.path.join(new_face_dir,face_name_align)
				try:
					faces_found += 1
					new_face_name.append(new_path)
				except:
					continue
		frame = frame[face_found_mask].reset_index(drop=True)
		frame['face_name_align'] = new_face_name
		print("{} out of {} faces are found in new dir!".format(faces_found, initial_rows))

	image_name_frame = frame['image_name'].apply(split_image_name)
	image_names = image_name_frame.unique()
	np.random.seed(42)
	image_names = np.random.permutation(image_names)

	n_images = len(image_names)
	n_train = int(train_pct * n_images)
	n_val = int((n_images-n_train)/2)
	n_test = n_images - n_train - n_val

	image_names_train = image_names[0:n_train]
	image_names_val = image_names[n_train:n_train+n_val]
	image_names_test = image_names[n_train+n_val:]

	print("{} images: {} training, {} validation, {} test".format(n_images, len(image_names_train), len(image_names_val), len(image_names_test)))

	train_frame = frame[image_name_frame.isin(image_names_train)].reset_index(drop=True)
	val_frame = frame[image_name_frame.isin(image_names_val)].reset_index(drop=True)
	test_frame = frame[image_name_frame.isin(image_names_test)].reset_index(drop=True)

	return {'train':train_frame, "val":val_frame, "test":test_frame, "all":frame}

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
class ImgAugTransform:
    
    def __init__(self):
        self.aug = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            #iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.05))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            iaa.Affine(
                scale={"x": (1, 1.1), "y": (1, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
                rotate=(-15, 15), # rotate by -15 to +15 degrees
                shear=(-8, 8), # shear by -8 to +8 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=['edge'] # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 0.1),
                            n_segments=(50, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 0.3), strength=(0, 2)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.3)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.3), direction=(0.0, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.02), per_channel=0.5),
                        #iaa.CoarseDropout(
                        #    (0.03, 0.15), size_percent=(0.02, 0.05),
                        #    per_channel=0.2
                        #),
                    ]),

                    # Invert each image's chanell with 5% probability.
                    # This sets each pixel value v to 255-v.
                    #iaa.Invert(0.05, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-15, 15), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.75, 1.25), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.ContrastNormalization((0.75, 1.75), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    #sometimes(
                    #    iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=0.25)
                    #),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.005, 0.01)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    def __call__(self, img):
        
        img = np.array(img)
        return self.aug.augment_image(img)


class FaceDataset(Dataset):

    def __init__(self, data_frame, transform=None, col_used=['race','gender','age']):

        self.data_frame = data_frame
        self.transform = transform
        # First two rows in celeba csvs are image names
        # TODO: make all image names at the first two columns.
        if col_used is None:
            col_used = [2, data_frame.shape[1]]
        if isinstance(col_used[0], str):
            self.col_used = col_used
        else:
            self.col_used = [self.data_frame.columns[i] for i in range(col_used[0], col_used[1])]

    def __len__(self):
        
        return len(self.data_frame)

    def __getitem__(self, idx):
        #idx is index from dataset
        #This is a mapping from your data_frame to the output of the mode
        img_name = self.data_frame.loc[idx, 'face_name_align']
        labels = []
        for col in self.col_used:
            labels.append(self.data_frame.loc[idx, col])

        # read image as ndarray, H*W*C
        image = dlib.load_rgb_image(img_name) 
        
        if self.transform:
            image = self.transform(image)
        
        # transform label to torch tensor
        # This sets the order of the label
        return (image, torch.from_numpy(np.asarray(labels)))


def make_datasets(train_frame, val_frame, give_dataloader=True, batch_size=64, col_used=None):
	device = torch.device('cuda:0')
	transform_train_data = transforms.Compose([
	    ImgAugTransform(),
	    lambda x: PIL.Image.fromarray(x),
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(), 
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	transformed_train_dataset = FaceDataset(data_frame=train_frame,
		                                   transform=transform_train_data,
                                                   col_used=col_used
		                                   )

	train_dataloader = DataLoader(transformed_train_dataset, batch_size=batch_size,
		                shuffle=True, num_workers=8)


	transform_test_data = transforms.Compose(([transforms.ToPILImage(), 
					       transforms.Resize((224, 224)),
		                              transforms.ToTensor(), 
		                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		                                  ]))

	transformed_test_dataset = FaceDataset(data_frame=val_frame,
		                                   transform=transform_test_data,
                                                   col_used=col_used
		                                   )

	test_dataloader = DataLoader(transformed_test_dataset, batch_size=batch_size,
		                shuffle=True, num_workers=8)
	if give_dataloader:
		return train_dataloader, test_dataloader
	else:
		return transformed_train_dataset, transformed_test_dataset
