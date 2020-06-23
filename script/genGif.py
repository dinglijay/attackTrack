import imageio
import glob
from os.path import join

if __name__ == "__main__":
    
    root_dir = './results'
    filenames = sorted(glob.glob(join(root_dir, '*.jp*')))

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimwrite(join(root_dir,'movie.gif'), images, duration=0.01)


    # # For longer movies, use the streaming approach:
    # with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
#         writer.append_data(image)