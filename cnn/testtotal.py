# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 01:44:14 2019

@author: cloh5
"""
from CGvsPhoto import Model
#database_path = '/Users/cloh5/Python_Scripts/FaceForensics/patchdatabase0/'
database_path = '/Users/cloh5/Python_Scripts/FaceForensics/patch100ObamaDFdatabase/'

# to change to the format of your image
image_size = 100

#clf2 = Model(database_path, image_size, config = 'Personal', filters = [32,32,64],
#            batch_size = 50, feature_extractor = 'Stats', remove_context = True, 
#            remove_filter_size = 5, only_green = False)

#test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/ObamaDFDatabase/'
#test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/Croppedobama_allmanipulated/Original_cropped/'
#test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/Croppedobama_allmanipulated/Deepfakes_cropped/'
test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/Croppedobama_allmanipulated/Face2Face_cropped/'
#test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/Croppedobama_allmanipulated/FaceSwap_cropped/'
#test_data_path = '/Users/cloh5/Python_Scripts/FaceForensics/Croppedobama_allmanipulated/NeuralTextures_cropped/'


#clfob100_pt.test_total_images(test_data_path = test_data_path,
#                      nb_images = 25, only_green = False, show_images=True, save_images =True, decision_rule = 'weighted_vote')

clfpt.test_total_images(test_data_path = test_data_path,
                      nb_images = 30, only_green = False, show_images=False, save_images =False, decision_rule = 'weighted_vote')