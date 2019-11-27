from CGvsPhoto import Model
# to change to your favorite database
database_path = '/Users/cloh5/Python_Scripts/FaceForensics/patch100ObamaDFdatabase/'


# to change to the format of your image
image_size = 100

# define a single-image classifier
clfpt = Model(database_path, image_size, config = 'Personal', filters = [32,32,64],
            batch_size = 50, feature_extractor = 'Stats', remove_context = True, 
            remove_filter_size = 5, only_green = False)


# trains the classifier and test it on the testing set
clfpt.train(nb_train_batch = 150,
          nb_test_batch = 80, 
          nb_validation_batch = 40,
          #validation_frequency = 20,
          show_filters = True)
