'''
generate gt file based on siammask tracking result
'''

file1 = open(join(args.base_path, 'groundtruth_rect.txt'), 'w') 
file1.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, w, h))

file1.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, x2-x, y2-y))
file1.close() 