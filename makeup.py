import cv2
import numpy as np
import tensorflow as tf
import torch
import utils.net as net
from blend_modes import darken_only, hard_light, normal
from PIL import Image
from torchvision import transforms
from utils.api import PRN
from utils.models import Segmentor
from utils.render import prepare_tri_weights, render_by_tri, render_texture
from utils.utils import de_norm, to_tensor, to_var


class Makeup:
    def __init__(self, args):
        # if args.pattern:
        self.pattern = Segmentor(args)
        self.pattern.test_model(args.checkpoint_pattern)
        self.color = net.Generator_branch(64, 6).cuda()
        self.color.load_state_dict(torch.load(args.checkpoint_color))
        self.color.eval()
        if args.prn:
            self.prn = PRN(is_dlib=True)

    def get_mask(self, img):
        x_tensor = to_tensor(img / 255)
        x_tensor = torch.from_numpy(x_tensor).unsqueeze(0).cuda()
        pr_mask = self.pattern.model.predict(x_tensor)
        pr_mask = pr_mask[0, 0, :, :].detach().cpu().numpy()
        return pr_mask

    def makeup(self, img_A, img_B):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img_A = transform(Image.fromarray(img_A))
        img_B = transform(Image.fromarray(img_B))
        img_A = img_A[None, :, :, :]
        img_B = img_B[None, :, :, :]
        real_org = to_var(img_A).cuda()
        real_ref = to_var(img_B).cuda()

        # Get makeup result
        fake_A, fake_B = self.color(real_org, real_ref)
        result = de_norm(fake_A.detach()[0]).cpu().numpy().transpose(1, 2, 0)
        result = cv2.resize(result, (256, 256), cv2.INTER_CUBIC)
        return result

    def prn_process(self, face):
        # --- face
        self.face = cv2.resize(face, (256, 256))
        self.pos = self.prn.process(self.face)
        self.vertices = self.prn.get_vertices(self.pos)
        #         self.face = face/255
        self.h, self.w, _ = self.face.shape
        self.triangles = self.prn.triangles
        vis_colors = np.ones((self.vertices.shape[0], 1))
        face_mask = render_texture(self.vertices.T, vis_colors.T, self.triangles.T, self.h, self.w, c=1)
        self.face_mask = np.squeeze(face_mask > 0).astype(np.float32)
        self.weights, self.dst_triangle_buffer = prepare_tri_weights(self.vertices.T, self.triangles.T, self.h, self.w)

        uv_face_eye = np.array(Image.open("./PRNet/uv-data/uv_face_eyes.png"))[:, :, :3] / 255
        new_colors = self.prn.get_colors_from_texture(uv_face_eye)
        new_colors = (new_colors > 0).astype("uint8")
        mask_out_eye = render_by_tri(
            new_colors.T,
            self.triangles.T,
            self.weights,
            self.dst_triangle_buffer,
            self.h,
            self.w,
            c=3,
        )
        self.mask_out_eye = (mask_out_eye > 0).astype("uint8")  # [:, :, np.newaxis]
        tf.reset_default_graph()

    def prn_process_target(self, face):
        # --- face
        face = cv2.resize(face, (256, 256))
        pos = self.prn.process(face)
        vertices = self.prn.get_vertices(pos)
        #         self.face = face/255
        h, w, _ = face.shape
        triangles = self.prn.triangles
        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = render_texture(vertices.T, vis_colors.T, triangles.T, h, w, c=1)
        face_mask = np.squeeze(face_mask > 0).astype(np.float32)
        weights, dst_triangle_buffer = prepare_tri_weights(vertices.T, triangles.T, h, w)
        # uv_face = cv2.imread("./PRNet/uv-data/uv_face.png")[:, :, 0] / 255.0
        texture = cv2.remap(
            face,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        tf.reset_default_graph()
        return texture

    def get_texture(self):
        texture = cv2.remap(
            self.face,
            self.pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        return texture

    def get_seg(self):
        texture = cv2.remap(
            self.face_seg,
            self.pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        return texture

    def render_texture(self, texture, patt_only=False):
        new_colors = self.prn.get_colors_from_texture(texture)
        new_image = render_by_tri(
            new_colors.T,
            self.triangles.T,
            self.weights,
            self.dst_triangle_buffer,
            self.h,
            self.w,
            c=3,
        )
        # new_face = self.face_mask[:, :, np.newaxis]*new_image + (1-self.face_mask[:, :, np.newaxis])*self.face/255
        # if patt_only:
        #     return new_face
        # else:
        return new_image

    def blend_imgs(self, source, reference, blend_mode="normal", alpha=0.8):
        """"""
        # blurred_mask = cv2.GaussianBlur(np.stack([self.face_mask, self.face_mask, self.face_mask], axis=2)*255, (25, 25), 0)
        blurred_mask = cv2.GaussianBlur(self.mask_out_eye * 255, (25, 25), 0)
        extend_mask = self.mask_out_eye.copy() * 255
        gray = cv2.cvtColor(extend_mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_mask = np.where(extend_mask == np.array([255, 255, 255]), blurred_mask, extend_mask)

        refer = reference * self.mask_out_eye + source * (1 - self.mask_out_eye)

        source = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_RGB2RGBA)
        source[:, :, 3] = np.ones((256, 256)) * 255

        refer = cv2.cvtColor(refer.astype("uint8"), cv2.COLOR_RGB2RGBA)
        refer[:, :, 3] = new_mask[:, :, 0]
        if blend_mode == "normal":
            blended_img = normal(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        elif blend_mode == "darken_only":
            blended_img = darken_only(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        elif blend_mode == "hard_light":
            blended_img = hard_light(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        # Image.fromarray(np.concatenate([source.astype('uint8'), blended_img[:, :, :3], tar.astype('uint8')], axis=1))
        return blended_img[:, :, :3]

    def get_blur_mask(self, source_seg):
        seg = cv2.resize(source_seg, (256, 256), interpolation=cv2.INTER_NEAREST)
        if len(seg.shape) == 3:
            seg = seg[:, :, 0]
        facial_mask = (seg == 1) + (seg == 2) + (seg == 3) + (seg == 6) + (seg == 7) + (seg == 9)
        facial_mask = facial_mask.astype("uint8")
        facial_mask = cv2.dilate(facial_mask, np.ones((10, 10), np.uint8), iterations=1)
        #         mask_out_eye = cv2.dilate(self.mask_out_eye, np.ones((10, 10),np.uint8), iterations = 1)
        facial_mask = (facial_mask == 1) * (self.mask_out_eye[:, :, 0] == 1)
        facial_mask = np.stack([facial_mask, facial_mask, facial_mask], axis=2).astype("uint8")
        blurred_mask = cv2.GaussianBlur(facial_mask * 255, (25, 25), 0)
        extend_mask = facial_mask.copy() * 255
        gray = cv2.cvtColor(extend_mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extend_mask = cv2.drawContours(extend_mask, contours, -1, (255, 255, 255), 30)
        new_mask = np.where(extend_mask == np.array([255, 255, 255]), blurred_mask, facial_mask)
        return new_mask

    def blend_imgs_2(self, source, reference, source_seg, blend_mode="normal", alpha=0.9):
        """"""
        # seg = np.array(Image.open(list_segs_A[0]))
        new_mask = self.get_blur_mask(source_seg)
        refer = reference * self.mask_out_eye + source * (1 - self.mask_out_eye)
        source = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_RGB2RGBA)
        source[:, :, 3] = np.ones((256, 256)) * 255

        refer = cv2.cvtColor(refer.astype("uint8"), cv2.COLOR_RGB2RGBA)
        refer[:, :, 3] = new_mask[:, :, 0]
        if blend_mode == "normal":
            blended_img = normal(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        elif blend_mode == "darken_only":
            blended_img = darken_only(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        elif blend_mode == "hard_light":
            blended_img = hard_light(source.astype("float"), refer.astype("float"), alpha).astype("uint8")
        # Image.fromarray(np.concatenate([source.astype('uint8'), blended_img[:, :, :3], tar.astype('uint8')], axis=1))
        return blended_img[:, :, :3]

    def location_to_crop(self, mini=False):
        if mini:
            idx = np.where(self.mask_out_eye[:, :, 0] == 1)
            x2, y2, x1, y1 = (
                np.min(idx[1]),
                np.max(idx[1]),
                np.min(idx[0]),
                np.max(idx[0]),
            )
            max_idx = np.argmax([y1 - x1, y2 - x2]) + 1

            if max_idx == 1:
                y2 = x2 + (y1 - x1)
            else:
                y1 = x1 + (y2 - x2)
            return x2, y2, x1, y1
        else:
            idx = np.where(self.mask_out_eye[:, :, 0] == 1)
            x2, y2, x1, y1 = (
                np.min(idx[1]),
                np.max(idx[1]),
                np.min(idx[0]),
                np.max(idx[0]),
            )
            return x2, y2, x1, y1
