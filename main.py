import os, cv2, random
from PIL import Image
import numpy as np
from parser import get_args
from makeup import Makeup
import time

def color_makeup(A_txt, B_txt, alpha):
	color = model.makeup(A_txt, B_txt)
	color = model.render_texture(rs)
	color = model.blend_imgs(model.face, rs*255, alpha=alpha)
	return color

def pattern_makeup(A_txt, B_txt, return_mask = False):
	mask = model.get_mask(B_txt)
	mask = (mask>0.0001).astype('uint8')
	pattern = A_txt*(1-mask)[:, :, np.newaxis] + B_txt*mask[:, :, np.newaxis]
	pattern = model.render_texture(pattern)
	pattern = model.blend_imgs(model.face, pattern, alpha=1)
	if return_mask:
		return pattern, mask
	else:
		return pattern

if __name__ == '__main__':
	args = get_args()
	model = Makeup(args)

	imgA = np.array(Image.open(args.input))
	imgB = np.array(Image.open(args.style))
	imgB = cv2.resize(imgB, (256, 256))

	model.prn_process(imgA)
	A_txt = model.get_texture()
	B_txt = model.prn_process_target(imgB)

	if args.color_only:
		output = color_makeup(A_txt, B_txt, args.alpha)
	elif args.pattern_only:
		output = pattern_makeup(A_txt, B_txt)
	else:
		color = color_makeup(A_txt, B_txt, args.alpha)
		pattern, mask = pattern_makeup(A_txt, B_txt, return_mask = True)
		output = color*(1-mask[:, :, np.newaxis])+pattern*mask[:, :, np.newaxis]
	x2, y2, x1, y1 = model.location_to_crop()
	output = np.concatenate([imgB[x2:], model.face[x2:], output[x2:]], axis=1)
	Image.fromarray((output).astype('uint8')).save(os.path.join(args.savedir, 'result.png'))