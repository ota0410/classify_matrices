import os,config,h5py
from PIL import Image
import random
import numpy as np
def get_files_list(image_path):
	file_list = []
	for subdir, dirs, files in os.walk(image_path):
		for file in files:
			file_path = os.path.join(subdir,file)
			file_list.append(file_path)
	return file_list
def get_train_test_file_path_split(file_list):
	file_list = np.array(file_list)
	np.random.shuffle(file_list)
	divide_indices = int(file_list.shape[0] *0.8)
	train_file_list = file_list[0:divide_indices]
	test_file_list = file_list[divide_indices:]
	train_file_list = list(train_file_list)
	test_file_list = list(test_file_list)
	# with open(json_file,'w') as f:
	# 	json.dump({'train_file_list' : train_file_list, 'test_file_list' : test_file_list},f)
	return train_file_list, test_file_list

def load_train_test_file_list(json_file):
	with open(json_file,'r')  as f:
		train_test = json.load(f)
		train_file_list = train_test['train_file_list']
		test_file_list = train_test['test_file_list']
	return train_file_list

def read_image(image_path):
	img = Image.open(image_path,'r')
	return img


def resize_img(resize_dim,img):
        resized_img = img.resize((resize_dim,resize_dim))
#	resized_img = img.resize(img, resize_dim, interpolation = cv2.INTER_LINEAR )
	return resized_img

def get_data(file_list,label_y_dict):
	x = np.zeros((len(file_list),224,224))
	y = np.zeros((len(file_list),1))
	for i in range(len(file_list)):
		file = file_list[i]
		img = read_image(file)
#               print (img)
		grey_image = img
		resized_image = resize_img(224,grey_image)
                resized_image.mode = "L"
		x[i] = resized_image
		y_k = os.path.basename(file).split('_')[0]
		y[i] = label_y_dict[y_k]
	# x = np.array(x,dtype = 'float32')
	y = np.array(y,dtype='uint8').reshape((len(y),1))
	return x,y
def load_sample_dataset():
	train_test_file_path = config.train_test_file_path
	with  h5py.File(train_test_file_path,'r') as hf:
		data_x = np.array(hf.get('data_x'))
		label_y = np.array(hf.get('label_y'))
		#test_x = np.array(hf.get('test_x'))
		#test_y = np.array(hf.get('test_y'))
	return data_x, label_y
def main():
	img_row,img_col = config.img_row, config.img_col
	img_path = config.img_path
	train_test_file_path = config.train_test_file_path
	label_y_dict = config.label_y_dict
	file_list = get_files_list(img_path)	
	random.shuffle(file_list)
	#Train_file_list ,test_file_list = get_train_test_file_path_split(file_list)
	data_x,label_y = get_data(file_list,label_y_dict)
	#test_x,test_y = get_data(test_file_list,label_y_dict)
	print(data_x.shape,label_y.shape)
	with h5py.File(train_test_file_path, 'w') as hf:
		 hf.create_dataset('data_x', data = data_x)
		 hf.create_dataset('label_y', data = label_y)
		 #hf.create_dataset('test_x', data = test_x)
		 #hf.create_dataset("test_y", data = test_y)
	
#if __name__ == '__main__':
	#pass
main()
	# train_x,train_y,test_x,test_y = load_sample_dataset()
	# print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
	# print train_x
	# print train_y




