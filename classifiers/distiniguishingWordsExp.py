
import utilities

if __name__ == "__main__":
    from pylab import *

    source_training_file = 'data/target.txt'
    source_auxiliary_file = 'data/target_auxiliary.txt'
    source_validation_file = 'data/source_validation.txt'

    source_training_data = genfromtxt(source_training_file, delimiter=',')
    source_auxiliary_data = genfromtxt(source_auxiliary_file, delimiter=',')
    source_validation_data = genfromtxt(source_validation_file, delimiter=',')

    top_dimensions = utilities.listTopDimensions(source_training_data[0:43,1:],source_training_data[0:43,0])
    print top_dimensions
    top_dimensions = utilities.listTopDimensions(source_auxiliary_data[0:8,1:],source_auxiliary_data[0:8,0])
    print top_dimensions
    top_dimensions = utilities.listTopDimensions(source_validation_data[0:11,1:],source_validation_data[0:11,0])
    print top_dimensions
    top_dimensions = utilities.listTopDimensions(source_training_data[:,1:],source_training_data[:,0])
    print top_dimensions
    top_dimensions = utilities.listTopDimensions(source_auxiliary_data[:,1:],source_auxiliary_data[:,0])
    print top_dimensions
    top_dimensions = utilities.listTopDimensions(source_validation_data[:,1:],source_validation_data[:,0])
    print top_dimensions