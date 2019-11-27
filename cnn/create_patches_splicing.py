from CGvsPhoto import Database_loader

# directory with the original database
source_db = '/Users/cloh5/Python_Scripts/FaceForensics/ObamaDFDatabase/'

# wanted size for the patches 
image_size = 100

# directory to store the patch database
target_patches = '/Users/cloh5/Python_Scripts/FaceForensics/patch100ObamaDFdatabase/'


# create a database manager 
DB = Database_loader(source_db, image_size, 
                     only_green=False, rand_crop = True)

# export a patch database    
DB.export_database(target_patches, 
                   nb_train = 3000, 
                   nb_test = 700, 
                   nb_validation = 300)

# directory to store splicing images 
# target_splicing = '/home/nicolas/Database/splicing2/'


# DB.export_splicing(target_splicing, 50)
