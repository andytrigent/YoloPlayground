import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import os
from PIL import Image

favicon = Image.open("favicon.png")
st.set_page_config(
    page_title="GenAI Demo | Trigent AXLR8 Labs",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Logo
logo_html = """
<style>
    [data-testid="stSidebarNav"] {
        background-image: url(https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png);
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 80%;
    }
</style>
"""
# Streamlit app layout
st.title(f"Financial News Sentiment Analysis")

st.sidebar.markdown(logo_html, unsafe_allow_html=True)

st.write(
    """

    🌟 **Damage Detection Simplified** 🌟

Dive into the world of advanced object detection with our solution, powered by the incredible [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) and [SAM](https://docs.ultralytics.com/models/sam/) models. These models are not just any ordinary models; they're specially trained using the vast and diverse [Roboflow](https://public.roboflow.com/) dataset, ensuring unparalleled accuracy and performance.

Join us on this exciting journey to unlock the full potential of your data with our state-of-the-art solution! 🚀


    """
)

# remove file in a folder ----------------------------------------------------------------------
folder = 'op_detection'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Save File ------------------------------------------------------------------------------------------


def save_uploadedfile(uploadedfile):
    with open(os.path.join("ip_image", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


# Model ----------------------------------------------------------------------------------------------------
image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
if image_file is not None:
    file_details = {"FileName": image_file.name, "FileType": image_file.type}

    conf_threshold = st.slider(
        'Confidence Threshold', min_value=0.0, max_value=1.0, value=0.35)

    if st.button('RUN'):

        st.write(file_details)
        save_uploadedfile(image_file)

        col1, col2, col3 = st.columns(3)

        image = cv2.imread("ip_image/" + image_file.name)

        with col1:
            st.text("Raw Image")
            st.image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Todo: Development

        from super_gradients.training import models

        # define class name
        class_names = ['crack_and_hole', 'medium_deformation', 'severe_deformation',
                       'severe_scratch', 'slight_deformation', 'slight_scratch', 'windshield_damage']

        # Todo: Get the model

        device = 'cuda' if torch.cuda.is_available() else "cpu"
        model_nas = models.get('yolo_nas_l',
                               num_classes=7,
                               checkpoint_path='nas_weight/ckpt_best.pth')

        # Todo: Object detection prediction
        model_nas.predict(image, conf=conf_threshold).save('op_detection')

        with col2:
            st.text("Detection Output")
            st.image('op_detection/pred_0.jpg')

        # Todo: Get BBOX
        model_pred = list(model_nas.predict(
            image, conf=conf_threshold)._images_prediction_lst)

        bboxes_xyxy = model_pred[0].prediction.bboxes_xyxy.tolist()

        # Todo: SAM

        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate(
                    [np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(coords, labels, ax, marker_size=375):
            pos_points = coords[labels == 1]
            neg_points = coords[labels == 0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                       marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                       marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

        def show_box(box, ax):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h,
                         edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        sam_checkpoint = "sam_weight/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = 'cuda' if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)

        # Todo: SAM predictor
        predictor.set_image(image)

        tensor_box = torch.tensor(bboxes_xyxy, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(
            tensor_box, image.shape[:2])

        batch_masks, batch_scores, batch_logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        for mask in batch_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        plt.axis('off')
        plt.savefig('my_image.jpg')

        with col3:
            st.text("Masked Output")
            st.image('my_image.jpg')

# Footer
footer_html = """
<div style="text-align: right; margin-right: 10%;">
    <p>
        Copyright © 2024, Trigent Software, Inc. All rights reserved. | 
        <a href="https://www.facebook.com/TrigentSoftware/" target="_blank">Facebook</a> |
        <a href="https://www.linkedin.com/company/trigent-software/" target="_blank">LinkedIn</a> |
        <a href="https://www.twitter.com/trigentsoftware/" target="_blank">Twitter</a> |
        <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank">YouTube</a>
    </p>
</div>
"""

# Custom CSS to make the footer sticky
footer_css = """
<style>
.footer {
    position: fixed;
    z-index: 1000;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
[data-testid="stSidebarNavItems"] {
    max-height: 100%!important;
}
</style>
"""

# Combining the HTML and CSS
footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

# Rendering the footer
st.markdown(footer, unsafe_allow_html=True)
