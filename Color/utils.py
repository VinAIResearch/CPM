import torch
from torch.autograd import Variable
from ops.histogram_matching import histogram_matching

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def rebound_box(mask_A, mask_B, mask_A_face):
    index_tmp = mask_A.nonzero()
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    mask_A_temp = mask_A.copy_(mask_A)
    mask_B_temp = mask_B.copy_(mask_B)
    mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                        mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
    mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                        mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
    mask_A_temp = to_var(mask_A_temp, requires_grad=False)
    mask_B_temp = to_var(mask_B_temp, requires_grad=False)
    return mask_A_temp, mask_B_temp

def mask_preprocess(mask_A, mask_B):
    index_tmp = mask_A.nonzero()
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    mask_A = to_var(mask_A, requires_grad=False)
    mask_B = to_var(mask_B, requires_grad=False)
    index = [x_A_index, y_A_index, x_B_index, y_B_index]
    index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
    return [mask_A, mask_B, index, index_2]

def get_mask(mask_A, mask_B, part = 'lips'):
	if part=='lips':
		mask_A_lip = (mask_A==7).float() + (mask_A==9).float()
		mask_B_lip = (mask_B==7).float() + (mask_B==9).float()
		mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = mask_preprocess(mask_A_lip, mask_B_lip)
		return mask_A_lip, mask_B_lip, index_A_lip, index_B_lip
	elif part=='skin':
		mask_A_skin = (mask_A==1).float() + (mask_A==6).float().float()
		mask_B_skin = (mask_B==1).float() + (mask_B==6).float().float()
		mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = mask_preprocess(mask_A_skin, mask_B_skin)
		return mask_A_skin, mask_B_skin, index_A_skin, index_B_skin
	elif part=='eye':
		mask_A_eye_left = (mask_A==4).float()
		mask_A_eye_right = (mask_A==5).float()
		mask_B_eye_left = (mask_B==4).float()
		mask_B_eye_right = (mask_B==5).float()
		mask_A_face = (mask_A==1).float() + (mask_A==6).float()
		mask_B_face = (mask_B==1).float() + (mask_B==6).float()
		# avoid the situation that images with eye closed
		if not ((mask_A_eye_left>0).any() and (mask_B_eye_left>0).any() and (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
			return Exception('No')
		mask_A_eye_left, mask_A_eye_right = rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
		mask_B_eye_left, mask_B_eye_right = rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
		mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = mask_preprocess(mask_A_eye_left, mask_B_eye_left)
		mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = mask_preprocess(mask_A_eye_right, mask_B_eye_right)
		return mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right, mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left
	else:
		print('Error!')

