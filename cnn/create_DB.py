from CGvsPhoto import construct_DB

# Change to the source of real images
source_real = "/Users/cloh5/Python_Scripts/FaceForensics/Obama_crop/Real/"
# Change to the source of CG images
source_CG = "/Users/cloh5/Python_Scripts/FaceForensics/Obama_crop/CGG/"
# The directory where the database will be saved
target_dir = '/Users/cloh5/Python_Scripts/FaceForensics/ObamaDFDatabase/'

# Construct a database with an equilibrated CG/Real ratio
# Formatted to be used with the image_loader
construct_DB(source_real = source_real, 
			  source_CG = source_CG,
			  target_dir = target_dir, 
			  nb_per_class = 390,
			  validation_proportion = 0.1, 
			  test_proportion = 0.2)
