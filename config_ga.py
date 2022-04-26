from colossalai.amp import AMP_TYPE

tst_batch_size = 5
inverse_order = True #0: [input, target], 1 - [target, input]
input_size = 256
ngf = 64
ndf = 64
lrG = 0.0002
lrD = 0.0002
beta1 = 0.5 #'beta1 for Adam optimizer'
beta2 = 0.999 #beta2 for Adam optimizer
resize_scale = 286 #'resize scale (0 is false)'
crop_size = 256 #crop size (0 is false)
fliplr = True #random fliplr True or False
L1_lambda = 100

BATCH_SIZE = 1
NUM_EPOCHS = 10
gradient_accumulation = 4