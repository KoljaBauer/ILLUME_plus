import os
import sys
sys.path.append(os.getcwd())
from torch.utils.data import Dataset

# print(os.getcwd())

# from diffusion.utils import load_model_inference
from torchvision.transforms.functional import to_pil_image, to_tensor
from einops import rearrange, repeat
import hydra
import random
import math
from PIL import Image, ImageDraw
import torch
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange
from collections import defaultdict
from ILLUME.imagenet_hierarchy import hierarchy_attr_to_classes, direct_subclasses, id_to_syn_name, super_class_dir, n_str_mapping
import torchvision.transforms as T
import webdataset as wds
import numpy as np
from torch.utils.data import DataLoader
import os
import accelerate
from accelerate import Accelerator
from einops import rearrange
import timm
import torch
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import glob
import json
import math
import scipy.stats
import argparse
import tarfile
import io
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from torchvision.transforms import functional as F


superclasses_21k = {
'bird.n.01' : ['animal.n.01'],
'crocodilian_reptile.n.01': ['reptile.n.01', 'animal.n.01'],
'butterfly.n.01': ['insect.n.01', 'animal.n.01'],
'snake.n.01': ['animal.n.01'],
'insect.n.01': ['animal.n.01'],
'fish.n.01': ['animal.n.01'],
'reptile.n.01': ['animal.n.01'],
'shark.n.01': ['fish.n.01', 'animal.n.01'],
'arachnid.n.01': ['animal.n.01'],
'lizard.n.01': ['reptile.n.01', 'animal.n.01'],
'primate.n.02': ['animal.n.01'],
'dog.n.01': ['animal.n.01'],
'feline.n.01': ['animal.n.01'],
'cat.n.01': ['feline.n.01', 'animal.n.01'],

}

imgnet_21k_class_dict = {'arachnid.n.01': ['n01780142',
                   'n01771766',
                   'n01779629',
                   'n01776313',
                   'n01781570',
                   'n01776705',
                   'n01778621',
                   'n01780426',
                   'n01773157',
                   'n01777909'],
 'bird.n.01': ['n02029706',
               'n01796519',
               'n01852400',
               'n01585715',
               'n01798839',
               'n02005399',
               'n01844231',
               'n01586374',
               'n01579028',
               'n01556182'],
 'butterfly.n.01': ['n02276749',
                    'n02280649',
                    'n02283077',
                    'n02282903',
                    'n02280458',
                    'n02282553',
                    'n02275560',
                    'n02274822',
                    'n02281787',
                    'n02276902'],
 'cat.n.01': ['n02125689',
              'n02126139',
              'n02125010',
              'n02125311',
              'n02121620',
              'n02126028',
              'n02123159',
              'n02124484',
              'n02127052',
              'n02127678'],
 'crocodilian_reptile.n.01': ['n01698640',
                              'n01697611',
                              'n01697178',
                              'n01698434',
                              'n01697749',
                              'n01699040',
                              'n01697457',
                              'n01696633',
                              'n01699675',
                              'n01698782'],
 'dog.n.01': ['n02090622',
              'n02097209',
              'n02097967',
              'n02084732',
              'n02094721',
              'n02085019',
              'n02093056',
              'n02090379',
              'n02109150',
              'n02099429'],
 'feline.n.01': ['n02125689',
                 'n02126139',
                 'n02125010',
                 'n01323068',
                 'n02125311',
                 'n02121620',
                 'n02120997',
                 'n02126028',
                 'n02123159',
                 'n02124484'],
 'fish.n.01': ['n02552171',
               'n02644665',
               'n01480516',
               'n02578454',
               'n02643836',
               'n01449374',
               'n01449712',
               'n02625612',
               'n01497413',
               'n02586543'],
 'insect.n.01': ['n02186153',
                 'n02251775',
                 'n02276749',
                 'n02172761',
                 'n02164464',
                 'n02252226',
                 'n02167151',
                 'n02301935',
                 'n02234570',
                 'n02248887'],
 'lizard.n.01': ['n01678043',
                 'n01680813',
                 'n01687665',
                 'n01682172',
                 'n01688243',
                 'n01686609',
                 'n01692523',
                 'n01677747',
                 'n01689411',
                 'n01680264'],
 'primate.n.02': ['n02494383',
                  'n02473983',
                  'n02486657',
                  'n02492660',
                  'n02483092',
                  'n02471300',
                  'n02487847',
                  'n02477782',
                  'n02498153',
                  'n02477187'],
 'reptile.n.01': ['n01733957',
                  'n01678043',
                  'n01748686',
                  'n01680813',
                  'n01734808',
                  'n01730812',
                  'n01687665',
                  'n01682172',
                  'n01731277',
                  'n01688243'],
 'shark.n.01': ['n01484562',
                'n01492708',
                'n01483830',
                'n01486010',
                'n01491361',
                'n01493146',
                'n01490112',
                'n01495006',
                'n01488038',
                'n01489501'],
 'snake.n.01': ['n01733957',
                'n01748686',
                'n01734808',
                'n01730812',
                'n01731277',
                'n01745902',
                'n01748389',
                'n01730185',
                'n01755740',
                'n01742447']}


class ImgNet21kDataset(Dataset):
    def __init__(self, tar_files, max_images=50, transform=None):
        """
        Args:
            tar_files (list): List of tar file paths.
            max_images (int): Maximum number of images to load per tar file.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.tar_files = tar_files
        self.max_images = max_images
        self.transform = T.Compose([
                            T.ToTensor(),
                            T.Resize(256),
                            T.CenterCrop(256),
                            T.Lambda(lambda x: x * 2 - 1),
                            T.Lambda(lambda x: x.bfloat16()),
                        ])
        self.samples = self._index_tar_files()

    def _index_tar_files(self):
        """Indexes the tar files and stores (tar_file, image_name) pairs."""
        samples = []
        for tar_path in self.tar_files:
            with tarfile.open(tar_path, "r") as tar:
                # Filter and randomly select up to max_images
                members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg"))]
                selected_members = random.sample(members, min(len(members), self.max_images))
                samples.extend([(tar_path, m.name) for m in selected_members])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_path, image_name = self.samples[idx]
        with tarfile.open(tar_path, "r") as tar:
            f = tar.extractfile(image_name)
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
            image = self.transform(image)
            return os.path.splitext(os.path.basename(tar_path))[0], image

class ImageNetSketchDataset(Dataset):
    def __init__(self, root_path):
        self.image_paths = []
        self.transform =  T.Compose([
                            T.ToTensor(),
                            T.Resize(256),
                            T.CenterCrop(256),
                            T.Lambda(lambda x: x * 2 - 1),
                            T.Lambda(lambda x: x.bfloat16()),
                        ])

        # Scan all class directories
        for class_dir in glob.glob(root_path):
            wnid = os.path.basename(class_dir)
            class_name = n_str_mapping[wnid]
            # print(f"processing class {class_name}")

            # Collect all image paths with class label
            for img_file in os.listdir(class_dir):
                self.image_paths.append((os.path.join(class_dir, img_file), class_name))
                

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return class_name, image


def categorical_crossentropy(true_dist, pred_dist):
    # Ensure no zero values to avoid log(0), add a small epsilon
    epsilon = 1e-10
    pred_dist = np.clip(pred_dist, epsilon, 1.0)
    
    # Calculate categorical cross-entropy
    cross_entropy = -np.sum(true_dist * np.log(pred_dist))
    return cross_entropy

def compute_metrics(predicted_cls_count_per_q_cls, hierarchy_name):

    metrics = {'kl_div': 0.0, 'cat_ce': 0.0}

    for q_subcls in predicted_cls_count_per_q_cls.keys():

        total_count = sum(list(predicted_cls_count_per_q_cls[q_subcls].values()))
        pred_prob_dict = {img_cls: float(pred_ct)/total_count for img_cls, pred_ct in predicted_cls_count_per_q_cls[q_subcls].items()}

        all_leafs = hierarchy_attr_to_classes[hierarchy_name][q_subcls] if q_subcls in hierarchy_attr_to_classes[hierarchy_name] else [q_subcls]
        # Uniform distribution (each category has equal probability)
        desired_probs_dict = {leaf: 1.0 / len(all_leafs) for leaf in all_leafs}

        desired_classes_list = list( set( desired_probs_dict.keys()))
        print(desired_classes_list)

        pred_prob_list = [pred_prob_dict[key] if key in pred_prob_dict else 0.0 for key in desired_classes_list]
        desired_prob_list = [desired_probs_dict[key] for key in desired_classes_list]

        # Compute KL divergence
        kl_div = scipy.stats.entropy(desired_prob_list, pred_prob_list)  # KL(P || Q)

        cat_ce = categorical_crossentropy(true_dist=desired_prob_list, pred_dist=pred_prob_list)

        metrics['kl_div'] += kl_div
        metrics['cat_ce'] += cat_ce

        print(f"Metrics for q_subcls {q_subcls}", flush=True)
        print(f"KL Divergence: {kl_div:.4f}", flush=True)
        print(f"Categorical Cross-Entropy: {kl_div:.4f}", flush=True)

    metrics['kl_div'] = metrics['kl_div'] / len(predicted_cls_count_per_q_cls.keys())
    metrics['cat_ce'] = metrics['cat_ce'] / len(predicted_cls_count_per_q_cls.keys())

    return metrics


def id_to_synset_name(id: str):
    offset = int(id[1:])
    return wn.synset_from_pos_and_offset("n", offset).name()

def generate_images_illume(cls, imgs_per_attr, hierarchy_name, set_direction_attr, bs, ctx_images, q_images, model, n_ctx, base_seed, device, use_21k_ctx, use_21k_q, img_res: int =256):

    generated_images_per_q_cls = defaultdict(list)
    direct_subclasses_qt = direct_subclasses[hierarchy_name][set_direction_attr]
    num_iters = math.ceil(imgs_per_attr // bs)
    queries_per_iter = bs
    print(f"num_iters is {num_iters} ; queries_per_iter is {queries_per_iter} ; bs is {bs}", flush=True)

    prompt_und = 'What is a common feature between these four images? Answer concisely in a few words.'

    inference_config_understanding = model.prepare_inference_config(
                                temperature=1.0,
                                top_k=50,
                                top_p=1.0,
                            )
    
    unconditional_prompt = model.default_editing_unconditional_template.format(resolution_tag='')
    inference_config_gen = model.prepare_inference_config(
        temperature=1.0,
        top_k=128,
        top_p=1.0,
        
        llm_cfg_scale = 1.5,
        diffusion_cfg_scale=1.5,
        diffusion_num_inference_steps=50,
        
        image_semantic_temperature= 0.7,
        image_semantic_top_k = 512,
        image_semantic_top_p = 0.8,
        unconditional_prompt=unconditional_prompt,
        resolution=(img_res,img_res)  # the resolution will be obtrain from the source image within the code.
    )

    for iter in trange(num_iters):
        context_set = build_context_set(rel_attr=set_direction_attr, images=ctx_images, 
                                        hierarchy_name=hierarchy_name, n_ctx=4, use_21k_ctx=use_21k_ctx).to(device=device, dtype=torch.bfloat16) # 1 c n_ctx h w"
        
        # Tile the context set
        context_set = rearrange(context_set, "1 c (row col) h w -> c (row h) (col w)", row=2, col=2).float() # range[-1, 1]
        print(f"{context_set.shape=} ; {context_set.min()=} ; {context_set.max()=}", flush=True)
        # convert to PIL images
        context_set = F.to_pil_image((context_set + 1) / 2)
        context_set.show()

        qt_leaf_classes = [cls] * queries_per_iter
        random.seed(base_seed + iter)
        query_list = [random.choice(q_images[leaf_cls]) for leaf_cls in qt_leaf_classes] # list of (c, h, w) tensors
        print(f"{query_list[0].max()=} ; {query_list[0].min()=}", flush=True)
        query_list_pil = [F.to_pil_image((q.float() + 1) / 2) for q in query_list]  # convert to PIL images
        # query = torch.cat([rearrange(q, " c h w -> 1 c 1 h w") for q in query_list], dim=0).to(device=device, dtype=torch.bfloat16)
        query_list_pil[0].show()
        query_list_pil[1].show()
        print(f"{len(query_list_pil)} ; {query_list_pil[0].size=}", flush=True)

        n_qs = len(query_list)

        # context_set_broadcast = repeat(context_set, "1 c n_ctx h w -> n_qs c n_ctx h w", n_qs=n_qs).to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            batch_data = [
                dict(prompt=prompt_und, images_data=[context_set])
            ] * bs
            # Infer common feature of context set
            outputs = model.inference_mllm(
                            batch_data, inference_config_understanding,
                            is_img_gen_task=False,  #  Remember set this for image understanding. 
                            do_sample=False  # You could add more params for the model.generate.
                        )
            
            common_features = [outputs[i]['output_text'] for i in range(len(outputs))]
            print(f"common features: {common_features}", flush=True)
            instructions = [f'Take the common feature from the query image and generate a new image with it. The feature you inferred as common is: {comm_feat}.' for comm_feat in common_features]
            print(f"{instructions=}", flush=True)

            prompts = [model.default_editing_template.format(resolution_tag='', content=instruction) for instruction in instructions]
            batch_data = [dict(prompt=prompts[i], images_data=[query_list_pil[i]]) for i in range(bs)]
            # Generate image tokens
            outputs = model.inference_mllm(batch_data, inference_config_gen, is_img_gen_task=True)

            # print(f"{outputs=}", flush=True)
            print(f"{len(outputs[0]['image_embed_inds'])=}", flush=True)
            # # Decode image tokens Using sdxl diffusion decoder
            # out_images = model.inference_tokenizer_decoder(outputs, inference_config_gen, use_diffusion_decoder=True)

            print(f"{outputs[0]['image_sizes']=}", flush=True)
            print(f"Starting to decode generated tokens ...")

            #  using vq tokenizer to decode image.
            out_images = model.inference_tokenizer_decoder(outputs, inference_config_gen, use_diffusion_decoder=False)

            # padded_image= convert_np_to_pil_img(out_images, outputs)[0]
            # generated_image = model.unpad_and_resize_back(padded_image, outputs[0]['original_sizes'][0], outputs[0]['original_sizes'][1])


        for n_q in range(queries_per_iter):
            sample_n_q = out_images[n_q]
            print(f"{sample_n_q.shape=} ; {sample_n_q.min()=} ; {sample_n_q.max()=}", flush=True)
            # F.to_pil_image(rearrange(sample_n_q, "h w c -> c h w")).show()
            # F.to_pil_image(sample_n_q).show()

            sample_n_q_tensor = torch.from_numpy(sample_n_q).float() / 255.0  # Convert to tensor and normalize to range [0, 1]
            print(f"{sample_n_q_tensor.shape=} ; {sample_n_q_tensor.min()=} ; {sample_n_q_tensor.max()=} ; {sample_n_q_tensor.dtype=}", flush=True)
            
            generated_images_per_q_cls[qt_leaf_classes[n_q]].append(sample_n_q_tensor)

    return generated_images_per_q_cls

def generate_images_vis_prompt(cls, imgs_per_attr, hierarchy_name, set_direction_attr, bs, ctx_images, q_images, model, n_ctx, base_seed, device, use_21k_ctx, use_21k_q):
    generated_images_per_q_cls = defaultdict(list)
    # all_subclasses_imgnet = hierarchy_attr_to_classes[hierarchy_name][set_direction_attr]
    direct_subclasses_qt = direct_subclasses[hierarchy_name][set_direction_attr]

    num_iters = math.ceil(imgs_per_attr // bs)
    queries_per_iter = bs

    print(f"num_iters is {num_iters} ; queries_per_iter is {queries_per_iter} ; bs is {bs}", flush=True)

    for iter in trange(num_iters):
        context_set = build_context_set(rel_attr=set_direction_attr, images=ctx_images, 
                                        hierarchy_name=hierarchy_name, n_ctx=4, use_21k_ctx=use_21k_ctx).to(device=device, dtype=torch.bfloat16)

        qt_leaf_classes = [cls] * queries_per_iter
        query_list = [random.choice(q_images[leaf_cls]) for leaf_cls in qt_leaf_classes]
        query = torch.cat([rearrange(q, " c h w -> 1 c 1 h w") for q in query_list], dim=0).to(device=device, dtype=torch.bfloat16)

        n_qs = len(query_list)

        context_set_broadcast = repeat(context_set, "1 c n_ctx h w -> n_qs c n_ctx h w", n_qs=n_qs).to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            z = torch.randn((n_qs,4,32,32), device=device, dtype=torch.bfloat16, generator=torch.Generator(device).manual_seed(base_seed + iter))

            print(f"z shape: {z.shape} ; context_set_broadcast shape: {context_set_broadcast.shape} ; query shape: {query.shape}")
            res = model.sample(
                                z=z,
                                context_set=context_set_broadcast,
                                query=query,
                            )
        print(f"res shape is {res.shape}")

        for n_q in range(queries_per_iter):
            sample_n_q = res[n_q, :, :, :]
            generated_images_per_q_cls[qt_leaf_classes[n_q]].append(sample_n_q)

    return generated_images_per_q_cls

def generate_images(cls, imgs_per_attr, hierarchy_name, set_direction_attr, bs, ctx_images, q_images, model, n_ctx, base_seed, device, use_21k_q, use_21k_ctx):
    generated_images_per_q_cls = defaultdict(list)
    # all_subclasses_imgnet = hierarchy_attr_to_classes[hierarchy_name][set_direction_attr]
    direct_subclasses_qt = direct_subclasses[hierarchy_name][set_direction_attr]

    num_iters = math.ceil(imgs_per_attr // bs)
    queries_per_iter = imgs_per_attr // num_iters

    print(f"cls is {cls}; num_iters is {num_iters} ; queries_per_iter is {queries_per_iter} ; bs is {bs}", flush=True)

    print(f"ctx_images image keys: {ctx_images.keys()}")
    print(f"q image keys: {q_images.keys()}")

    for iter in trange(num_iters):
        context_set = build_context_set(rel_attr=set_direction_attr, images=ctx_images,  use_21k_ctx=use_21k_ctx,
                                        hierarchy_name=hierarchy_name, n_ctx=n_ctx).to(device=device, dtype=torch.bfloat16)

        # qt_sub_classes = random.choices(direct_subclasses_qt, k=queries_per_iter)
        # qt_leaf_classes = [random.choice(hierarchy_attr_to_classes[hierarchy_name][qt_sub] if qt_sub in hierarchy_attr_to_classes[hierarchy_name] else [qt_sub]) for qt_sub in qt_sub_classes]
        qt_leaf_classes = [cls] * queries_per_iter
        query_list = [random.choice(q_images[leaf_cls]) for leaf_cls in qt_leaf_classes]
        query = torch.cat([rearrange(q, " c h w -> 1 c 1 h w") for q in query_list], dim=2).to(device=device, dtype=torch.bfloat16)
        n_qs = len(query_list)

        print(f"query shape: {query.shape}")

        with torch.no_grad():
            z = torch.randn((1, n_qs, 4, 32, 32), device=device, dtype=torch.bfloat16, generator=torch.Generator(device).manual_seed(base_seed + iter))
            sampling_mode = ['fix'] 
            attr_def_ordered = ['variation']

            print(f"z shape: {z.shape} ; context_set shape: {context_set.shape} ; query shape: {query.shape}")

            res, _ = model.sample(
                                        z=z,
                                        context_set=context_set,
                                        query=query,
                                        attr_def=attr_def_ordered,
                                        sampling_mode=sampling_mode,
                                        broadcast_noise=False,
                                    )
        print(f"res shape is {res.shape}")
        for n_q in range(queries_per_iter):
            sample_n_q = res[0, :, n_q, :, :]
            generated_images_per_q_cls[qt_leaf_classes[n_q]].append(sample_n_q)

    return generated_images_per_q_cls

def build_context_set(rel_attr, images, hierarchy_name: str, n_ctx: int, use_21k_ctx: bool = False):
    if use_21k_ctx:
        if rel_attr in imgnet_21k_class_dict:
            possible_set_classes = imgnet_21k_class_dict[rel_attr]
            set_clss = random.sample(possible_set_classes, k=min(len(possible_set_classes), n_ctx) )
            if len(possible_set_classes) > n_ctx:
                set_clss += random.choices(possible_set_classes, k=n_ctx - len(possible_set_classes))
        else: # animal direction
            set_super_clss = random.sample(list(imgnet_21k_class_dict.keys()), k=n_ctx)
            set_clss = [random.choice(imgnet_21k_class_dict[set_super_cls]) for set_super_cls in set_super_clss]
        
    else:
        set_super_clss = random.sample(direct_subclasses[hierarchy_name][rel_attr], k=min(len(direct_subclasses[hierarchy_name][rel_attr]), n_ctx)) 

        if len(direct_subclasses[hierarchy_name][rel_attr]) < n_ctx:
            set_super_clss += random.choices(direct_subclasses[hierarchy_name][rel_attr], k=n_ctx - len(direct_subclasses[hierarchy_name][rel_attr])) 

        set_clss = [np.random.choice(hierarchy_attr_to_classes[hierarchy_name][set_super_cls] if set_super_cls in hierarchy_attr_to_classes[hierarchy_name] else [set_super_cls]) for set_super_cls in set_super_clss]
    print(f"set_clss: {set_clss} ; use_21k_ctx: {use_21k_ctx}")
    set_imgs = torch.stack([random.choice(images[cls]) for cls in set_clss], dim=1)

    return rearrange(set_imgs, "c n_ctx h w -> 1 c n_ctx h w")

def get_classifications(generated_images_per_q_cls, classifier, bs):
    # Define the image transformation (same as used in ViT training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predicted_cls_count_per_q_cls = {q_cls: {'top_1': defaultdict(int), 'top_5': defaultdict(int)} for q_cls in generated_images_per_q_cls.keys()}

    for q_cls, gen_imgs_list in generated_images_per_q_cls.items():
        print(f"Processing generated images for query class {q_cls} ; {len(gen_imgs_list)=} ; {gen_imgs_list[0].shape=}", flush=True)

        input_tensors_transformed = [transform(rearrange(gen_img, "h w c -> c h w").float()) for gen_img in gen_imgs_list]
        input_tensor = torch.stack(input_tensors_transformed, dim=0)
        # input_tensor = transform(torch.cat([rearrange(gen_img, "c h w -> 1 c h w") for gen_img in gen_imgs_list], dim=0)).float()

        print(f"classifier input shape: {input_tensor.shape}", flush=True)
                                                       
        image_chunks = torch.chunk(input_tensor, chunks=(input_tensor.shape[0] + bs - 1) // bs, dim=0)

        for im_chunk in image_chunks:
            print(f"im_chunk shape: {im_chunk.shape}", flush=True)

            with torch.no_grad():
                output = classifier(im_chunk)
            
            predicted_classes = output.argmax(dim=1)
            
            for pred_cls in predicted_classes:
                predicted_cls_count_per_q_cls[q_cls]['top_1'][id_to_syn_name[pred_cls.item()]] += 1

            top5_pred_classes = torch.topk(output, k=5, dim=1).indices
            for preds in top5_pred_classes:
                for pred_cls in preds:  # Iterate over top-5 predictions
                    predicted_cls_count_per_q_cls[q_cls]['top_5'][id_to_syn_name[pred_cls.item()]] += 1
        
        print(f"Predicted class counts for {q_cls}: {predicted_cls_count_per_q_cls[q_cls]}")
    
    return predicted_cls_count_per_q_cls

def process_attr(cls, model, ctx_images, q_images, hierarchy_name, bs: int, classifier, device, out_path: str, model_type: str, imgs_per_attr: int = 100, n_ctx=5, 
                 base_seed=123, use_21k_ctx: bool = False, use_21k_q: bool = False, do_classification: bool = True, pregenerated_imgs: bool = False):
    if use_21k_q:
        next_super = next((k for k, v in imgnet_21k_class_dict.items() if cls in v))
        super_classes = superclasses_21k[next_super] + [next_super] 
    else:
        super_classes = super_class_dir[hierarchy_name][cls]
    generated_images_set_dir_q_cls = defaultdict()

    print(f" ==== PROCESSING CLASS {cls} ========")

    if not pregenerated_imgs:
        print(f"Generating images for class {cls} with model {model_type}", flush=True)

        for set_direction_attr in super_classes:
            if use_21k_ctx and set_direction_attr not in list(list(imgnet_21k_class_dict.keys()) + ['animal.n.01']):
                print(f"Skipping set_direction {set_direction_attr} because no suitable context set can be built from loaded ImgNet21k classes")
                continue

            # Generate 100 images per set direction and class
            if model_type == "set_learner":
                generated_images_per_q_cls = generate_images(cls=cls, imgs_per_attr=imgs_per_attr, hierarchy_name=hierarchy_name, 
                                set_direction_attr=set_direction_attr, bs=bs, ctx_images=ctx_images, q_images=q_images, model=model, n_ctx=n_ctx, base_seed=base_seed, 
                                device=device, use_21k_ctx=use_21k_ctx, use_21k_q=use_21k_q)
            
            elif model_type == "vis_prompt":
                generated_images_per_q_cls = generate_images_vis_prompt(cls=cls, imgs_per_attr=imgs_per_attr, hierarchy_name=hierarchy_name, 
                                set_direction_attr=set_direction_attr, bs=bs, ctx_images=ctx_images, q_images=q_images, model=model, n_ctx=n_ctx, base_seed=base_seed, 
                                device=device, use_21k_ctx=use_21k_ctx, use_21k_q=use_21k_q)

            elif model_type == "illume":
                generated_images_per_q_cls = generate_images_illume(cls=cls, imgs_per_attr=imgs_per_attr, hierarchy_name=hierarchy_name, 
                                set_direction_attr=set_direction_attr, bs=bs, ctx_images=ctx_images, q_images=q_images, model=model, n_ctx=n_ctx, base_seed=base_seed, 
                                device=device, use_21k_ctx=use_21k_ctx, use_21k_q=use_21k_q)
            
            generated_images_set_dir_q_cls[set_direction_attr] = generated_images_per_q_cls

        print(f"Finished generating images for class {cls}", flush=True)

        torch.save(generated_images_set_dir_q_cls, os.path.join(out_path, f"{cls}_gen_imgs_per_dir_per_q.pth"))
        print("Saved generated images successfully!", flush=True)  
    else:
        print(f"Loading pregenerated images for class {cls}", flush=True)
        # generated_images_set_dir_q_cls = torch.load(os.path.join(out_path, f"{cls}_gen_imgs_per_dir_per_q.pth"))
        with open(os.path.join(out_path, f"{cls}_gen_imgs_per_dir_per_q.pth"), 'rb') as f:
            generated_images_set_dir_q_cls = torch.load(f, weights_only=False, map_location=device)
        
        print("Loaded pregenerated images successfully!", flush=True)  

    if do_classification:
        print("Starting classification", flush=True)

        print(f"superclasses: {super_classes} ; generated_images_set_dir_q_cls.keys(): {generated_images_set_dir_q_cls.keys()}")

        predicted_cls_count_per_cls_per_q = defaultdict()

        for set_direction_attr in super_classes:
            if set_direction_attr == "turtle":
                continue
            predicted_cls_count_per_q_cls = get_classifications(generated_images_set_dir_q_cls[set_direction_attr], classifier=classifier, bs=bs)
            predicted_cls_count_per_cls_per_q[set_direction_attr] = predicted_cls_count_per_q_cls


        out_file_path = os.path.join(out_path, f"{cls}_predicted_cls_count_per_dir_per_q.json")
        print(f" ====== Writing output for cls {cls} to {out_file_path} =========")
        with open(out_file_path, "w") as f:
            json.dump(dict(predicted_cls_count_per_cls_per_q), f, indent=4)

        print("Saved classification results successfully!", flush=True)    
    else:
        print("Skipping classification", flush=True)



def load_model_from_config_inference(path, device):
    if isinstance(path, str):
        path = Path(path)
    
    ckpt_path = path / "checkpoints"
    if (ckpt_path / "latest").exists():
        with open(ckpt_path / "latest", mode="r") as f:
            tag = f.readlines()[0]
    else:
        tag = sorted(glob.glob(str(ckpt_path / "step_*")))[-1]
    print(f"Latest tag for {ckpt_path}: {tag}")

    cfg: DictConfig = OmegaConf.load(path / ".hydra" / "config.yaml")

    OmegaConf.resolve(cfg)
    cfg_model = cfg.model
    model = hydra.utils.instantiate(cfg_model)
    model.eval()
    model.load_state_dict(
        torch.load(
            ckpt_path / tag / "inference.pt",
            map_location="cpu",
        ),
        strict=True,
    )
    model.to(torch.bfloat16)
    model.to(device)
    model.eval()
    return model, cfg


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath('.'), '../vision_tokenizer/')))
from generation_eval.models.builder import build_eval_model

def load_illume_model(device):
    model_name = 'ILLUME'

    mllm_config_path="../configs/example/illume_plus_3b/illume_plus_qwen2_5_3b_stage3.py"
    tokenizer_config_path="../configs/example/dualvitok/dualvitok_anyres_max512.py"
    vq_tokenizer_ckpt_path="../checkpoints/dualvitok/pytorch_model.bin"
    diffusion_decoder_path="../checkpoints/dualvitok-sdxl-decoder/"
    torch_dtype = 'fp16'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # local_rank = 0 if 'cuda' in device else -1

    eval_model_cfg = dict(
        type=model_name,
        config=mllm_config_path,
        tokenizer_config=tokenizer_config_path,
        diffusion_decoder_path=diffusion_decoder_path,
        tokenizer_checkpoint=vq_tokenizer_ckpt_path,
        torch_dtype=torch_dtype
    )
    logging.info(f'Building ILLUME model with config: {eval_model_cfg}')
    inference_engine = build_eval_model(eval_model_cfg)

    inference_engine.device = device # Overall device
    inference_engine.mllm_device = device
    inference_engine.vq_device = device
    inference_engine.diffusion_device = device
    return inference_engine


def main():
    parser = argparse.ArgumentParser(description="Provide model and data paths")
    parser.add_argument('--model_path', type=str, default="/export/scratch/ra49veb/set-learner/34474", help="Path to the model")
    parser.add_argument('--data_path', type=str, default="/export/scratch/ra63ral/data/imagenet_shards_raw_shuffled/imagenet_val_{000000..000006}.tar", help="Path to the data")
    parser.add_argument('--hierarchy_name', type=str, default="extended_v1", help="hierarchy_name ")
    parser.add_argument('--out_path', type=str, default="./", help="output path")
    parser.add_argument('--model_type', type=str, default="set_learner", help="model type")
    parser.add_argument('--ctx_imgs', type=str, default="ImageNet", help="what ctx_imgs images to use")
    parser.add_argument('--q_imgs', type=str, default="ImageNet", help="what query images to use")
    parser.add_argument('--sketch_path', type=str, default="/export/scratch/ra49veb/datasets/imagenet-sketch/sketch", help="path to sketch images")
    parser.add_argument('--imgnet21k_path', type=str, default="/export/group/datasets/winter21_whole", help="path to sketch images")
    parser.add_argument('--bs', type=int, default=100, help="batch size")
    parser.add_argument('--no_classification', action='store_false', dest='do_classification')
    parser.add_argument('--pregenerated_imgs', action='store_true', help="Whether to use pregenerated images")
    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Initialize Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index 
    world_size = accelerator.num_processes 

    bs = args.bs
    hierarchy_name = args.hierarchy_name

    all_animal_classes = list(hierarchy_attr_to_classes[hierarchy_name]['animal.n.01'])
    # all_relevant_attrs = list(hierarchy_attr_to_classes[hierarchy_name].keys())

    print(f" === Processing total of {len(all_animal_classes)} classes")

    animal_classes_rank = np.array_split(np.array(all_animal_classes), world_size)[rank]
    
    # animal_classes_rank = ['hognose_snake.n.01']
    print(f"On rank {rank}, got class list {animal_classes_rank}")

    # Instantiate model
    if args.model_type in ["set_learner", "vis_prompt"]:
        model, cfg = load_model_from_config_inference(args.model_path, device=accelerator.device)
        model.eval()
        model.c_dropout = 0.0 
        model.cfg_scale = 4.0 

    elif args.model_type == "illume":
        print(f"Using ILLUME model", flush=True)
        model = load_illume_model(device=accelerator.device)

    if args.model_type == 'set_learner':
        model.direction_in_proj.to(torch.float32)
    
    print(f"args.datapath is {args.data_path}")


    # Optionally load selected ImageNet21k classes into memory
    if args.ctx_imgs == "ImgNet21k" or args.q_imgs == "ImgNet21k":
        base_path = args.imgnet21k_path
        tar_files = [os.path.join(base_path, f"{n_str}.tar") for n_list in imgnet_21k_class_dict.values() for n_str in n_list]
        print("Loading selected parts of ImgNet21k into memory ...")
        dataset_21k_ds = ImgNet21kDataset(tar_files)

        imgnet_21k_dl = DataLoader(dataset_21k_ds, batch_size=32, num_workers=8, shuffle=True)

        imgnet_21k_images = defaultdict(list)
        for b in tqdm(imgnet_21k_dl):
            # print(f"iter, len(b['cls_name'] = {b['cls_name']})")
            for cls_name, t in zip(b[0], b[1]):
                imgnet_21k_images[cls_name].append(t)
        print("Before print")
        print([(class_name, len(loaded_imgs)) for class_name, loaded_imgs in imgnet_21k_images.items()] )
        print("after print")

    if args.ctx_imgs == "ImageNet" or args.q_imgs == "ImageNet":
        print("Loading ImgNet into memory ...")
        # imgnet_path = os.path.join(args.data_path, "imagenet_val_{000000..000006}.tar")
        # Load ImageNet val split into memory
        # ds = wds.WebDataset(imgnet_path, 
        #                     nodesplitter=None).decode('torch').map_dict(jpg = T.Compose([
        #         T.Resize(256),
        #         T.CenterCrop(256),
        #         T.Lambda(lambda x: x * 2 - 1),
        #         T.Lambda(lambda x: x.bfloat16()),
        #     ]
        #     ),).batched(32)

        # imgnet_dl = DataLoader(ds, num_workers=7, batch_size=None)

        imgnet_path = sorted(glob.glob(os.path.join(args.data_path, "imagenet_val_*.tar")))

        ds = wds.WebDataset(imgnet_path, 
                    nodesplitter=None).decode('torch').map_dict(jpg = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.Lambda(lambda x: x * 2 - 1),
        T.Lambda(lambda x: x.bfloat16()),
        ]
        ),)

        imgnet_dl = DataLoader(ds, num_workers=7, batch_size=32)

        # Fill up dictionary with images per class
        imgnet_images = defaultdict(list)
        for b in tqdm(imgnet_dl):
            # print(f"iter, len(b['cls_name'] = {b['cls_name']})")
            for cls_name, t in zip(b["cls_name"], b["jpg"]):
                imgnet_images[n_str_mapping[cls_name.decode("utf-8")]].append(t)

        print([(class_name, len(loaded_imgs)) for class_name, loaded_imgs in imgnet_images.items()] )

        print([(class_name, len(loaded_imgs)) for class_name, loaded_imgs in imgnet_images.items()] )

    if args.ctx_imgs == "Sketch" or args.q_imgs == "Sketch":
        print("Loading ImgNet Sketch into memory ...")
        sketch_dataset = ImageNetSketchDataset(os.path.join(args.sketch_path, "n*"))
        sketch_dataloader = DataLoader(sketch_dataset, batch_size=32, num_workers=8, shuffle=True)

        # Dictionary to store processed images
        sketch_images = defaultdict(list)

        # Iterate through DataLoader
        for class_names, img_tensors in tqdm(sketch_dataloader):
            for class_name, img_tensor in zip(class_names, img_tensors):
                sketch_images[class_name].append(img_tensor)

    # Optionally load ImageNet sketch into memory
    if args.ctx_imgs == "ImageNet":
        ctx_images = imgnet_images
    elif args.ctx_imgs == "Sketch":
        ctx_images = sketch_images
    elif args.ctx_imgs == "ImgNet21k":
        ctx_images = imgnet_21k_images
    else:
        print(f"Error: No valid ctx_images provided!")

    if args.q_imgs == "ImageNet":
        q_images = imgnet_images
    elif args.q_imgs == "Sketch":
        q_images = sketch_images
    elif args.q_imgs == "ImgNet21k":
        q_images = imgnet_21k_images
    else:
        print(f"Error: No valid q_images provided!")

    print(f"q_images keys: {q_images.keys()}")

    print(f"Using context images from {args.ctx_imgs} and query images from {args.q_imgs}")

    ### Load Classifier
    # Load the ViT-L/16 model pretrained on ImageNet-21K and fine-tuned on ImageNet-1K
    classifier = timm.create_model("vit_large_patch16_224", pretrained=True)
    classifier.eval()  # Set to evaluation mode
    classifier.to(device=accelerator.device)

    use_21k_ctx = args.ctx_imgs == "ImgNet21k"
    use_21k_q = args.q_imgs == "ImgNet21k"

    classes_to_process = animal_classes_rank 
    if use_21k_q:
        classes_to_process = [n_str for n_list in imgnet_21k_class_dict.values() for n_str in n_list]

    for animal_cls in tqdm(classes_to_process):
        print(f"Processing attr {animal_cls} ...")
        process_attr(cls=animal_cls, model=model, classifier=classifier, ctx_images=ctx_images, 
                     q_images=q_images, hierarchy_name=hierarchy_name, bs=bs, 
                     imgs_per_attr=100, n_ctx=5, base_seed=123, device=accelerator.device, 
                     out_path=args.out_path, model_type=args.model_type, use_21k_ctx=use_21k_ctx, 
                     use_21k_q=use_21k_q, do_classification=args.do_classification, pregenerated_imgs=args.pregenerated_imgs)


if __name__ == "__main__":
    main()
    